import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

abstract class EyelidClassificationLocalDataSource {
  Future<Map<String, dynamic>> predict(String imagePath);

  Future<void> dispose();
}

class EyelidClassificationLocalDataSourceImpl
    implements EyelidClassificationLocalDataSource {

  OrtSession? _session;

  final List<String> labelNames = [
    'eyelid',
    'normal',
  ];

  // ============================================================
  // Extract ONNX model/data files
  // ============================================================

  Future<String> _extractAssetToTemp(String assetKey) async {
    final byteData = await rootBundle.load(assetKey);

    final bytes = byteData.buffer.asUint8List(
      byteData.offsetInBytes,
      byteData.lengthInBytes,
    );

    final tempDir = await getTemporaryDirectory();

    final fileName = assetKey.split('/').last;

    final filePath =
        '${tempDir.path}${Platform.pathSeparator}$fileName';

    final file = File(filePath);

    await file.writeAsBytes(
      bytes,
      flush: true,
    );

    return filePath;
  }

  // ============================================================
  // Load model
  // ============================================================

  Future<void> _loadModel() async {
    const modelAsset =
        'assets/models/dinov2_best.onnx';

    const modelDataAsset =
        'assets/models/dinov2_best.onnx.data';

    final modelPath =
        await _extractAssetToTemp(modelAsset);

    await _extractAssetToTemp(modelDataAsset);

    final runtime = OnnxRuntime();

    _session = await runtime.createSession(
      modelPath,
      options: OrtSessionOptions(),
    );

    print('DINOv2 model loaded.');
  }

  // ============================================================
  // Image preprocessing
  // ============================================================

  Future<Float32List> _preprocessImage(
      String imagePath) async {

    final Uint8List bytes =
        await File(imagePath).readAsBytes();

    img.Image? image =
        img.decodeImage(bytes);

    if (image == null) {
      throw Exception(
        'Unable to decode image',
      );
    }

    // ----------------------------------------------------------
    // Resize to 224 x 224
    // ----------------------------------------------------------

    final img.Image resized =
        img.copyResize(
      image,
      width: 224,
      height: 224,
      interpolation:
          img.Interpolation.linear,
    );

    // ----------------------------------------------------------
    // ImageNet normalization
    // ----------------------------------------------------------

    const mean = [
      0.485,
      0.456,
      0.406,
    ];

    const std = [
      0.229,
      0.224,
      0.225,
    ];

    // ----------------------------------------------------------
    // Create CHW tensor
    //
    // Shape:
    // [3, 224, 224]
    // ----------------------------------------------------------

    final input =
        Float32List(
      3 * 224 * 224,
    );

    int index = 0;

    // ----------------------------------------------------------
    // RED
    // ----------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {

        final pixel =
            resized.getPixel(x, y);

        final r =
            pixel.r.toDouble() / 255.0;

        input[index++] =
            (r - mean[0]) / std[0];
      }
    }

    // ----------------------------------------------------------
    // GREEN
    // ----------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {

        final pixel =
            resized.getPixel(x, y);

        final g =
            pixel.g.toDouble() / 255.0;

        input[index++] =
            (g - mean[1]) / std[1];
      }
    }

    // ----------------------------------------------------------
    // BLUE
    // ----------------------------------------------------------

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {

        final pixel =
            resized.getPixel(x, y);

        final b =
            pixel.b.toDouble() / 255.0;

        input[index++] =
            (b - mean[2]) / std[2];
      }
    }

    return input;
  }

  // ============================================================
  // Flatten ONNX output
  // ============================================================

  List<double> _flattenOutput(dynamic data) {

    final List<double> result = [];

    void flatten(dynamic value) {

      if (value is num) {

        result.add(
          value.toDouble(),
        );

      } else if (value is List) {

        for (final item in value) {
          flatten(item);
        }

      } else if (value is Float32List) {

        for (final item in value) {
          result.add(
            item.toDouble(),
          );
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

  // ============================================================
  // Softmax
  // ============================================================

  List<double> _softmax(
      List<double> logits) {

    if (logits.isEmpty) {
      return [];
    }

    final maxLogit =
        logits.reduce(math.max);

    final expValues =
        logits.map(
      (x) => math.exp(
        x - maxLogit,
      ),
    ).toList();

    final sum =
        expValues.reduce(
      (a, b) => a + b,
    );

    return expValues.map(
      (x) => x / sum,
    ).toList();
  }

  // ============================================================
  // Prediction
  // ============================================================

  @override
  Future<Map<String, dynamic>> predict(
      String imagePath) async {

    // ----------------------------------------------------------
    // Total prediction timer
    //
    // This measures:
    // preprocessing
    // + tensor creation
    // + ONNX inference
    // + output processing
    //
    // Model loading is intentionally NOT included.
    // ----------------------------------------------------------

    final totalStopwatch =
        Stopwatch()..start();

    try {

      // --------------------------------------------------------
      // Load model if not already loaded
      // --------------------------------------------------------

      if (_session == null) {
        await _loadModel();
      }

      // --------------------------------------------------------
      // Preprocess image
      // --------------------------------------------------------

      final inputData =
          await _preprocessImage(
        imagePath,
      );

      // --------------------------------------------------------
      // Input information
      // --------------------------------------------------------

      final inputName =
          _session!.inputNames.first;

      final inputShape = [
        1,
        3,
        224,
        224,
      ];

      // --------------------------------------------------------
      // Create ONNX tensor
      // --------------------------------------------------------

      final inputOrt =
          await OrtValue.fromList(
        inputData,
        inputShape,
      );

      final inputs = {
        inputName: inputOrt,
      };

      // --------------------------------------------------------
      // PURE ONNX INFERENCE TIMER
      // --------------------------------------------------------

      final inferenceStopwatch =
          Stopwatch()..start();

      final outputs =
          await _session!.run(
        inputs,
      );

      inferenceStopwatch.stop();

      final inferenceTimeMs =
          inferenceStopwatch.elapsedMicroseconds /
              1000.0;

      // --------------------------------------------------------
      // Get output
      // --------------------------------------------------------

      final outputName =
          _session!.outputNames.first;

      final outputOrt =
          outputs[outputName];

      if (outputOrt == null) {

        await inputOrt.dispose();

        for (final value
            in outputs.values) {
          await value.dispose();
        }

        throw Exception(
          'Model output is null',
        );
      }

      // --------------------------------------------------------
      // Read output
      // --------------------------------------------------------

      final rawOutput =
          await outputOrt.asList();

      // --------------------------------------------------------
      // Convert output to logits
      // --------------------------------------------------------

      final logits =
          _flattenOutput(
        rawOutput,
      );

      // --------------------------------------------------------
      // Softmax
      // --------------------------------------------------------

      final scores =
          _softmax(
        logits,
      );

      if (scores.isEmpty) {

        await inputOrt.dispose();

        for (final value
            in outputs.values) {
          await value.dispose();
        }

        throw Exception(
          'Empty model output',
        );
      }

      // --------------------------------------------------------
      // Find predicted class
      // --------------------------------------------------------

      int predictedIndex = 0;

      for (
        int i = 1;
        i < scores.length;
        i++
      ) {

        if (
          scores[i] >
          scores[predictedIndex]
        ) {
          predictedIndex = i;
        }
      }

      // --------------------------------------------------------
      // Label
      // --------------------------------------------------------

      final label =
          predictedIndex <
                  labelNames.length
              ? labelNames[
                  predictedIndex
                ]
              : 'Unknown';

      // --------------------------------------------------------
      // Confidence
      // --------------------------------------------------------

      final confidence =
          scores[predictedIndex];

      // --------------------------------------------------------
      // Dispose ONNX values
      // --------------------------------------------------------

      await inputOrt.dispose();

      for (final value
          in outputs.values) {
        await value.dispose();
      }

      // --------------------------------------------------------
      // Stop total timer
      // --------------------------------------------------------

      totalStopwatch.stop();

      final totalTimeMs =
          totalStopwatch.elapsedMicroseconds /
              1000.0;

      // --------------------------------------------------------
      // Print timing
      // --------------------------------------------------------

      print(
        'DINOv2 inference time: '
        '${inferenceTimeMs.toStringAsFixed(2)} ms',
      );

      print(
        'DINOv2 total prediction time: '
        '${totalTimeMs.toStringAsFixed(2)} ms',
      );

      // --------------------------------------------------------
      // Return result
      // --------------------------------------------------------

      return {
        'label': label,

        'confidence': confidence,

        'scores': scores,

        'output': rawOutput,

        'logits': logits,

        // Timing
        'inference_time_ms':
            inferenceTimeMs,

        'total_time_ms':
            totalTimeMs,
      };
    }

    catch (e) {

      totalStopwatch.stop();

      rethrow;
    }
  }

  // ============================================================
  // Dispose model
  // ============================================================

  @override
  Future<void> dispose() async {

    if (_session != null) {

      await _session!.close();

      _session = null;

      print(
        'DINOv2 model disposed.',
      );
    }
  }
}