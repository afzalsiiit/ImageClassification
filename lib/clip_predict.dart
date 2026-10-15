import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'bpe_tokenizer.dart';

// ============================================================
// CLIP CLASSIFIER INTERFACE
// ============================================================

abstract class ClipClassifier {
  Future<Map<String, dynamic>> predict(String imagePath);

  Future<void> dispose();
}

// ============================================================
// CLIP IMPLEMENTATION
// ============================================================

class ClipClassifierImpl implements ClipClassifier {
  OrtSession? _session;

  // ==========================================================
  // CLIP LABELS
  // ==========================================================

  final List<String> labels = [
    'a photo of eye with tumor',
    'a photo of eye without tumor',
  ];

  // ==========================================================
  // CLIP IMAGE NORMALIZATION
  // ==========================================================

  static const List<double> mean = [0.48145466, 0.4578275, 0.40821073];

  static const List<double> std = [0.26862954, 0.26130258, 0.27577711];

  // ==========================================================
  // EXTRACT ASSET TO TEMPORARY DIRECTORY
  // ==========================================================

  Future<String> _extractAssetToTemp(String assetKey) async {
    final byteData = await rootBundle.load(assetKey);

    final bytes = byteData.buffer.asUint8List(
      byteData.offsetInBytes,
      byteData.lengthInBytes,
    );

    final tempDir = await getTemporaryDirectory();

    final fileName = assetKey.split('/').last;

    final filePath = '${tempDir.path}${Platform.pathSeparator}$fileName';

    final file = File(filePath);

    await file.writeAsBytes(bytes, flush: true);

    return filePath;
  }

  // ==========================================================
  // LOAD CLIP MODEL
  // ==========================================================

  Future<void> _loadModel() async {
    const modelAsset = 'assets/models/clip_vit32.onnx';

    const modelDataAsset = 'assets/models/clip_vit32.onnx.data';

    print('Loading CLIP model...');

    // Extract ONNX model
    final modelPath = await _extractAssetToTemp(modelAsset);

    // Extract external data
    //
    // IMPORTANT:
    // Both files are extracted into the same directory.
    //
    await _extractAssetToTemp(modelDataAsset);

    final runtime = OnnxRuntime();

    _session = await runtime.createSession(
      modelPath,
      options: OrtSessionOptions(),
    );

    print('CLIP model loaded.');
    print('CLIP input names: ${_session!.inputNames}');

    print('CLIP output names: ${_session!.outputNames}');
  }

  // ==========================================================
  // IMAGE PREPROCESSING
  // ==========================================================

  Future<Float32List> _preprocessImage(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();

    final decodedImage = img.decodeImage(bytes);

    if (decodedImage == null) {
      throw Exception('Unable to decode image');
    }

    // --------------------------------------------------------
    // RGB
    // --------------------------------------------------------

    final image = img.bakeOrientation(decodedImage);

    // --------------------------------------------------------
    // Resize to 224 x 224
    // --------------------------------------------------------

    final resized = img.copyResize(
      image,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );

    // --------------------------------------------------------
    // CLIP tensor
    //
    // Shape:
    // [3, 224, 224]
    //
    // Batch is added later:
    // [1, 3, 224, 224]
    // --------------------------------------------------------

    final input = Float32List(3 * 224 * 224);

    int index = 0;

    // --------------------------------------------------------
    // RED
    // --------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);

        final r = pixel.r.toDouble() / 255.0;

        input[index++] = (r - mean[0]) / std[0];
      }
    }

    // --------------------------------------------------------
    // GREEN
    // --------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);

        final g = pixel.g.toDouble() / 255.0;

        input[index++] = (g - mean[1]) / std[1];
      }
    }

    // --------------------------------------------------------
    // BLUE
    // --------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);

        final b = pixel.b.toDouble() / 255.0;

        input[index++] = (b - mean[2]) / std[2];
      }
    }

    return input;
  }

  // ==========================================================
  // TOKENIZE CLIP LABELS
  // ==========================================================

  Future<Int32List> _tokenizeLabels() async {
    // --------------------------------------------------------
    // Extract tokenizer.json from assets.
    //
    // CustomBPETokenizer expects a filesystem path.
    // --------------------------------------------------------

    const tokenizerAsset = 'assets/tokenizer.json';

    final tokenizerPath = await _extractAssetToTemp(tokenizerAsset);

    final tokenizer = CustomBPETokenizer(tokenizerPath, 77);

    final List<int> allTokenIds = [];

    // --------------------------------------------------------
    // Encode each CLIP prompt
    // --------------------------------------------------------

    for (final label in labels) {
      final tokenIds = tokenizer.encode(label);

      allTokenIds.addAll(tokenIds);
    }

    // --------------------------------------------------------
    // Expected shape:
    //
    // [2, 77]
    //
    // Two prompts:
    //
    // Prompt 0 = tumor
    // Prompt 1 = normal
    // --------------------------------------------------------

    if (allTokenIds.length != labels.length * 77) {
      throw Exception(
        'Invalid CLIP token tensor size. '
        'Expected ${labels.length * 77}, '
        'got ${allTokenIds.length}',
      );
    }

    return Int32List.fromList(allTokenIds);
  }

  // ==========================================================
  // SOFTMAX
  // ==========================================================

  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) {
      return [];
    }

    final maxLogit = logits.reduce(math.max);

    final expValues = logits
        .map((value) => math.exp(value - maxLogit))
        .toList();

    final sum = expValues.reduce((a, b) => a + b);

    return expValues.map((value) => value / sum).toList();
  }

  // ==========================================================
  // FLATTEN OUTPUT
  // ==========================================================

  List<double> _flattenOutput(dynamic data) {
    final List<double> result = [];

    void flatten(dynamic value) {
      if (value is num) {
        result.add(value.toDouble());
      } else if (value is Float32List) {
        for (final item in value) {
          result.add(item.toDouble());
        }
      } else if (value is List) {
        for (final item in value) {
          flatten(item);
        }
      } else {
        throw Exception(
          'Unsupported ONNX output type: '
          '${value.runtimeType}',
        );
      }
    }

    flatten(data);

    return result;
  }

  // ==========================================================
  // PREDICTION
  // ==========================================================

  @override
  Future<Map<String, dynamic>> predict(String imagePath) async {
    // ========================================================
    // TOTAL TIMER
    //
    // Includes:
    // - model loading if needed
    // - image preprocessing
    // - BPE tokenization
    // - ONNX inference
    // - result processing
    // ========================================================

    final totalWatch = Stopwatch()..start();

    // ========================================================
    // LOAD MODEL
    // ========================================================

    if (_session == null) {
      await _loadModel();
    }

    // ========================================================
    // IMAGE PREPROCESSING
    // ========================================================

    final imageData = await _preprocessImage(imagePath);

    // ========================================================
    // TEXT TOKENIZATION
    // ========================================================

    final textData = await _tokenizeLabels();

    // ========================================================
    // INPUT NAMES
    // ========================================================

    final inputNames = _session!.inputNames;

    if (!inputNames.contains('IMAGE') || !inputNames.contains('TEXT')) {
      throw Exception(
        'Expected CLIP inputs IMAGE and TEXT, '
        'but found $inputNames',
      );
    }

    // ========================================================
    // CREATE IMAGE ORT VALUE
    // ========================================================

    final imageOrt = await OrtValue.fromList(imageData, [1, 3, 224, 224]);

    // ========================================================
    // CREATE TEXT ORT VALUE
    //
    // Shape:
    // [2, 77]
    // ========================================================

    final textOrt = await OrtValue.fromList(textData, [2, 77]);

    // ========================================================
    // INFERENCE TIMER
    //
    // ONLY ONNX Runtime execution
    // ========================================================

    final inferenceWatch = Stopwatch()..start();

    final outputs = await _session!.run({'IMAGE': imageOrt, 'TEXT': textOrt});

    inferenceWatch.stop();

    // ========================================================
    // GET LOGITS PER IMAGE
    // ========================================================

    final outputOrt = outputs['LOGITS_PER_IMAGE'];

    if (outputOrt == null) {
      await imageOrt.dispose();
      await textOrt.dispose();

      for (final value in outputs.values) {
        await value.dispose();
      }

      throw Exception('LOGITS_PER_IMAGE output not found.');
    }

    // ========================================================
    // READ OUTPUT
    // ========================================================

    final rawOutput = await outputOrt.asList();

    final logits = _flattenOutput(rawOutput);

    print('CLIP logits: $logits');

    // ========================================================
    // CHECK OUTPUT
    // ========================================================

    if (logits.length != labels.length) {
      await imageOrt.dispose();
      await textOrt.dispose();

      for (final value in outputs.values) {
        await value.dispose();
      }

      throw Exception(
        'Unexpected CLIP output size. '
        'Expected ${labels.length}, '
        'got ${logits.length}',
      );
    }

    // ========================================================
    // SOFTMAX
    // ========================================================

    final probabilities = _softmax(logits);

    print(
      'CLIP probabilities: '
      '$probabilities',
    );

    // ========================================================
    // FIND BEST CLASS
    // ========================================================

    int predictedIndex = 0;

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[predictedIndex]) {
        predictedIndex = i;
      }
    }

    // ========================================================
    // RESULT
    // ========================================================

    final predictedLabel = labels[predictedIndex];

    final confidence = probabilities[predictedIndex];

    // ========================================================
    // STOP TOTAL TIMER
    // ========================================================

    totalWatch.stop();

    final inferenceTimeMs = inferenceWatch.elapsedMicroseconds / 1000.0;

    final totalTimeMs = totalWatch.elapsedMicroseconds / 1000.0;

    // ========================================================
    // DISPOSE INPUT/OUTPUT
    // ========================================================

    await imageOrt.dispose();

    await textOrt.dispose();

    for (final value in outputs.values) {
      await value.dispose();
    }

    // ========================================================
    // RETURN
    // ========================================================

    return {
      'label': predictedLabel,

      'confidence': confidence,

      'scores': probabilities,

      'logits': logits,

      // ONNX execution only
      'inference_time_ms': inferenceTimeMs,

      // Complete prediction operation
      'total_time_ms': totalTimeMs,
    };
  }

  // ==========================================================
  // DISPOSE MODEL
  // ==========================================================

  @override
  Future<void> dispose() async {
    if (_session != null) {
      print('Disposing CLIP model...');

      await _session!.close();

      _session = null;

      print('CLIP model disposed.');
    }
  }
}
