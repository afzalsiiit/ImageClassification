import 'dart:io';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import 'predict.dart';
import 'efficientnet_predict.dart';

void main() {
  runApp(const MyApp());
}


enum ModelType {
  dino,
  efficientNet,
  dinoApi,
  efficientNetApi,
  clipapi,
  gemmaapi
}


class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Eyelid Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

// ============================================================
// HOME PAGE
// ============================================================

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {

  // ==========================================================
  // IMAGE PICKER
  // ==========================================================

  final ImagePicker _picker = ImagePicker();

  // ==========================================================
  // LOCAL MODELS
  // ==========================================================

  final EyelidClassificationLocalDataSourceImpl _dino =
      EyelidClassificationLocalDataSourceImpl();

  final EfficientNetClassifierImpl _efficientNet =
      EfficientNetClassifierImpl();


  final String _apiBaseUrl =
      'http://192.168.137.1:8000';

  // ==========================================================
  // CURRENT MODEL
  // ==========================================================

  ModelType _selectedModel =
      ModelType.dino;

  // ==========================================================
  // IMAGE
  // ==========================================================

  File? _selectedImage;

  // ==========================================================
  // RESULT
  // ==========================================================

  String _prediction =
      'Select an image';

  String _confidence = '';

  // ==========================================================
  // TIMING
  // ==========================================================

  double? _inferenceTimeMs;

  double? _totalTimeMs;

  // ==========================================================
  // STATE
  // ==========================================================

  bool _isPredicting = false;

  bool _isSwitchingModel = false;

  // ==========================================================
  // MODEL NAME
  // ==========================================================

  String get _modelName {

    switch (_selectedModel) {

      case ModelType.dino:
        return 'DINOv2';

      case ModelType.efficientNet:
        return 'EfficientNet';

      case ModelType.dinoApi:
        return 'DINOv2 (API)';
      
      case ModelType.efficientNetApi:
        return 'EfficientNet (API)';

      case ModelType.clipapi:
        return 'Clip (API)';
      
      case ModelType.gemmaapi:
        return 'Gemma (API)';
    }
  }

  // ==========================================================
  // SWITCH MODEL
  // ==========================================================

  Future<void> _switchModel(
    ModelType newModel,
  ) async {

    // --------------------------------------------------------
    // Same model
    // --------------------------------------------------------

    if (newModel == _selectedModel) {
      return;
    }

    // --------------------------------------------------------
    // Do not switch during prediction
    // --------------------------------------------------------

    if (_isPredicting) {
      return;
    }

    setState(() {

      _isSwitchingModel = true;

      _prediction =
          'Switching model...';

      _confidence = '';

      _inferenceTimeMs = null;

      _totalTimeMs = null;
    });

    try {

      // ======================================================
      // DISPOSE CURRENT LOCAL MODEL
      // ======================================================

      if (_selectedModel ==
          ModelType.dino) {

        print(
          'Disposing DINOv2...',
        );

        await _dino.dispose();
      }

      else if (_selectedModel ==
          ModelType.efficientNet) {

        print(
          'Disposing EfficientNet...',
        );

        await _efficientNet.dispose();
      }

      // ======================================================
      // API MODEL
      // ======================================================

      // Nothing needs to be disposed when the
      // selected model is DINOv2 API because
      // the model is running on the FastAPI server.

      // ======================================================
      // CHANGE MODEL
      // ======================================================

      setState(() {

        _selectedModel =
            newModel;

        _prediction =
            '$_modelName selected';

        _confidence = '';

        _inferenceTimeMs = null;

        _totalTimeMs = null;
      });

      // ======================================================
      // IMPORTANT
      // ======================================================
      //
      // For local models:
      //
      // The new model is NOT loaded here.
      //
      // It will be loaded when Predict is pressed.
      //
      // This keeps memory usage low.
      //
      // ======================================================

    } catch (e) {

      setState(() {

        _prediction =
            'Model switch failed';

        _confidence =
            e.toString();
      });

    } finally {

      setState(() {

        _isSwitchingModel =
            false;
      });
    }
  }

  // ==========================================================
  // PICK IMAGE
  // ==========================================================

  Future<void> _pickImage(
    ImageSource source,
  ) async {

    final XFile? image =
        await _picker.pickImage(
      source: source,

      maxWidth: 1280,

      maxHeight: 1280,

      imageQuality: 100,
    );

    if (image == null) {
      return;
    }

    setState(() {

      _selectedImage =
          File(image.path);

      _prediction =
          'Image ready';

      _confidence = '';

      _inferenceTimeMs =
          null;

      _totalTimeMs =
          null;
    });
  }

  // ==========================================================
  // API PREDICTION
  // ==========================================================

  Future<Map<String, dynamic>> _predictUsingApi(
    String imagePath,
    String endpoint,
  ) async {

    // --------------------------------------------------------
    // Start total API timer
    // --------------------------------------------------------

    final stopwatch =
        Stopwatch()..start();

    // --------------------------------------------------------
    // API URL
    // --------------------------------------------------------

    final uri =
        Uri.parse(
      '$_apiBaseUrl/$endpoint',
    );

    print(
      'API URL: $uri',
    );

    // --------------------------------------------------------
    // Create multipart request
    // --------------------------------------------------------

    final request =
        http.MultipartRequest(
      'POST',
      uri,
    );

    // --------------------------------------------------------
    // Add image
    // --------------------------------------------------------

    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        imagePath,
      ),
    );

    // --------------------------------------------------------
    // Send request
    // --------------------------------------------------------

    final streamedResponse =
        await request.send();

    // --------------------------------------------------------
    // Convert streamed response
    // --------------------------------------------------------

    final response =
        await http.Response.fromStream(
      streamedResponse,
    );

    // --------------------------------------------------------
    // Stop timer
    // --------------------------------------------------------

    stopwatch.stop();

    final totalTimeMs =
        stopwatch.elapsedMicroseconds /
            1000.0;

    // --------------------------------------------------------
    // Check status
    // --------------------------------------------------------

    if (response.statusCode != 200) {

      throw Exception(
        'API error '
        '${response.statusCode}: '
        '${response.body}',
      );
    }

    // --------------------------------------------------------
    // Decode JSON
    // --------------------------------------------------------

    final data =
        json.decode(
      response.body,
    );

    // --------------------------------------------------------
    // Prediction
    // --------------------------------------------------------

    final prediction =
        data['prediction']
            .toString();

    // --------------------------------------------------------
    // Confidence
    // --------------------------------------------------------

    final confidence =
        (data['confidence'] as num).toDouble();


    // --------------------------------------------------------
    // Server inference time
    // --------------------------------------------------------

    final inferenceTimeMs = 

      data['inference_time_ms'] != null ? (data['inference_time_ms'] as num).toDouble() : 0.0;

    // --------------------------------------------------------
    // Return common result format
    // --------------------------------------------------------

    return {

      'label':
          prediction,

      'confidence':
          confidence,

      'inference_time_ms':
          inferenceTimeMs,

      'total_time_ms':
          totalTimeMs,
    };
  }

  // ==========================================================
  // PREDICT
  // ==========================================================

  Future<void> _predict() async {

    // --------------------------------------------------------
    // Check image
    // --------------------------------------------------------

    if (_selectedImage == null) {

      setState(() {

        _prediction =
            'Please select an image first';

        _confidence = '';

        _inferenceTimeMs =
            null;

        _totalTimeMs =
            null;
      });

      return;
    }

    // --------------------------------------------------------
    // Start prediction
    // --------------------------------------------------------

    setState(() {

      _isPredicting =
          true;

      _prediction =
          'Running $_modelName...';

      _confidence = '';

      _inferenceTimeMs =
          null;

      _totalTimeMs =
          null;
    });

    try {

      Map<String, dynamic> result;

      // ======================================================
      // DINOv2 LOCAL
      // ======================================================

      if (_selectedModel ==
          ModelType.dino) {

        print(
          'Using DINOv2 Local',
        );

        result =
            await _dino.predict(
          _selectedImage!.path,
        );
      }

      // ======================================================
      // EFFICIENTNET LOCAL
      // ======================================================

      else if (_selectedModel ==
          ModelType.efficientNet) {

        print(
          'Using EfficientNet Local',
        );

        result =
            await _efficientNet.predict(
          _selectedImage!.path,
        );
      }

      else if (_selectedModel ==
            ModelType.dinoApi) {

              result = await _predictUsingApi(_selectedImage!.path,
              'dino',);
            }
      else if (_selectedModel == ModelType.clipapi) {

        result = await _predictUsingApi(_selectedImage!.path, 'clip',);
      }

      else if (_selectedModel == ModelType.gemmaapi) {

        result = await _predictUsingApi(_selectedImage!.path, 'gemma');
      }

      // ======================================================
      // DINOv2 API
      // ======================================================

      else {


        result =
            await _predictUsingApi(
          _selectedImage!.path,
          'efficient',
        );
      }

      // ======================================================
      // RESULT
      // ======================================================

      final label =
          result['label']
              as String;

      final confidence =
          (result['confidence']
                  as num)
              .toDouble();

      // ======================================================
      // TIMING
      // ======================================================

      final inferenceTime =
          (result['inference_time_ms']
                  as num)
              .toDouble();

      final totalTime =
          (result['total_time_ms']
                  as num)
              .toDouble();

      // ======================================================
      // UPDATE UI
      // ======================================================

      setState(() {

        _prediction =
            label;

        _confidence =
            '${(confidence * 100).toStringAsFixed(2)}%';

        _inferenceTimeMs =
            inferenceTime;

        _totalTimeMs =
            totalTime;
      });

      // ======================================================
      // DEBUG
      // ======================================================

      print(
        '----------------------------------------',
      );

      print(
        'Model: $_modelName',
      );

      print(
        'Prediction: $label',
      );

      print(
        'Confidence: '
        '${(confidence * 100).toStringAsFixed(2)}%',
      );

      print(
        'Inference time: '
        '${inferenceTime.toStringAsFixed(2)} ms',
      );

      print(
        'Total prediction time: '
        '${totalTime.toStringAsFixed(2)} ms',
      );

      print(
        '----------------------------------------',
      );

    } catch (e) {

      print(
        'Prediction error: $e',
      );

      setState(() {

        _prediction =
            'Prediction failed';

        _confidence =
            e.toString();

        _inferenceTimeMs =
            null;

        _totalTimeMs =
            null;
      });

    } finally {

      setState(() {

        _isPredicting =
            false;
      });
    }
  }

  // ==========================================================
  // BUILD
  // ==========================================================

  @override
  Widget build(
    BuildContext context,
  ) {

    return Scaffold(

      // ========================================================
      // APP BAR
      // ========================================================

      appBar: AppBar(

        title:
            const Text(
          'Eyelid Classifier',
        ),

        centerTitle: true,
      ),

      // ========================================================
      // BODY
      // ========================================================

      body: SafeArea(

        child: Padding(

          padding:
              const EdgeInsets.all(16),

          child: Column(

            children: [

              // =================================================
              // MODEL SELECTOR
              // =================================================

              DropdownButtonFormField<ModelType>(

                value:
                    _selectedModel,

                decoration:
                    const InputDecoration(

                  labelText:
                      'Select Model',

                  border:
                      OutlineInputBorder(),
                ),

                items: const [

                  // ---------------------------------------------
                  // DINOv2 LOCAL
                  // ---------------------------------------------

                  DropdownMenuItem(

                    value:
                        ModelType.dino,

                    child:
                        Text(
                      'DINOv2 (Local)',
                    ),
                  ),

                  // ---------------------------------------------
                  // EFFICIENTNET LOCAL
                  // ---------------------------------------------

                  DropdownMenuItem(

                    value:
                        ModelType.efficientNet,

                    child:
                        Text(
                      'EfficientNet (Local)',
                    ),
                  ),

                  // ---------------------------------------------
                  // DINOv2 API
                  // ---------------------------------------------

                  DropdownMenuItem(

                    value:
                        ModelType.dinoApi,

                    child:
                        Text(
                      'DINOv2 (API)',
                    ),
                  ),

                  DropdownMenuItem(
                    value: ModelType.efficientNetApi,
                    child: Text("EfficientNet (API)")),
                  
                  DropdownMenuItem(
                    value: ModelType.clipapi,
                    child: Text("Clip (API)")),
                  
                  DropdownMenuItem(
                    value: ModelType.gemmaapi,
                    child: Text("Gemma (API)"))
                ],

                onChanged:
                    _isPredicting ||
                            _isSwitchingModel

                        ? null

                        : (value) {

                            if (value != null) {

                              _switchModel(
                                value,
                              );
                            }
                          },
              ),

              const SizedBox(
                height: 16,
              ),

              // =================================================
              // IMAGE
              // =================================================

              Expanded(

                child:
                    Container(

                  width:
                      double.infinity,

                  decoration:
                      BoxDecoration(

                    borderRadius:
                        BorderRadius.circular(
                      16,
                    ),

                    border:
                        Border.all(
                      color:
                          Colors.grey.shade300,
                    ),

                    color:
                        Colors.grey.shade100,
                  ),

                  child:

                      _selectedImage == null

                          ? const Center(

                              child:
                                  Column(

                                mainAxisAlignment:
                                    MainAxisAlignment.center,

                                children: [

                                  Icon(
                                    Icons.image,
                                    size: 80,
                                    color:
                                        Colors.grey,
                                  ),

                                  SizedBox(
                                    height: 16,
                                  ),

                                  Text(
                                    'No image selected',
                                  ),
                                ],
                              ),
                            )

                          : ClipRRect(

                              borderRadius:
                                  BorderRadius.circular(
                                16,
                              ),

                              child:
                                  Image.file(
                                _selectedImage!,

                                fit:
                                    BoxFit.contain,
                              ),
                            ),
                ),
              ),

              const SizedBox(
                height: 16,
              ),

              // =================================================
              // RESULT
              // =================================================

              Text(

                'Model: $_modelName',

                style:
                    const TextStyle(

                  fontSize: 16,

                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              const SizedBox(
                height: 6,
              ),

              // =================================================
              // PREDICTION
              // =================================================

              Text(

                _prediction,

                style:
                    const TextStyle(

                  fontSize: 24,

                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              // =================================================
              // CONFIDENCE
              // =================================================

              if (_confidence.isNotEmpty)

                Text(
                  'Confidence: $_confidence',
                ),

              // =================================================
              // INFERENCE TIME
              // =================================================

              if (_inferenceTimeMs != null)

                Text(

                  'Inference: '
                  '${_inferenceTimeMs!.toStringAsFixed(2)} ms',
                ),

              // =================================================
              // TOTAL TIME
              // =================================================

              if (_totalTimeMs != null)

                Text(

                  'Total: '
                  '${_totalTimeMs!.toStringAsFixed(2)} ms',
                ),

              const SizedBox(
                height: 12,
              ),

              // =================================================
              // CAMERA + GALLERY
              // =================================================

              Row(

                children: [

                  // ------------------------------------------------
                  // CAMERA
                  // ------------------------------------------------

                  Expanded(

                    child:
                        ElevatedButton.icon(

                      onPressed:
                          _isPredicting ||
                                  _isSwitchingModel

                              ? null

                              : () =>
                                  _pickImage(
                                ImageSource
                                    .camera,
                              ),

                      icon:
                          const Icon(
                        Icons.camera_alt,
                      ),

                      label:
                          const Text(
                        'Camera',
                      ),
                    ),
                  ),

                  const SizedBox(
                    width: 12,
                  ),

                  // ------------------------------------------------
                  // GALLERY
                  // ------------------------------------------------

                  Expanded(

                    child:
                        ElevatedButton.icon(

                      onPressed:
                          _isPredicting ||
                                  _isSwitchingModel

                              ? null

                              : () =>
                                  _pickImage(
                                ImageSource
                                    .gallery,
                              ),

                      icon:
                          const Icon(
                        Icons.photo,
                      ),

                      label:
                          const Text(
                        'Gallery',
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(
                height: 10,
              ),

              // =================================================
              // PREDICT BUTTON
              // =================================================

              SizedBox(

                width:
                    double.infinity,

                child:
                    ElevatedButton(

                  onPressed:
                      _isPredicting ||
                              _isSwitchingModel

                          ? null

                          : _predict,

                  child:

                      _isPredicting ||
                              _isSwitchingModel

                          ? const SizedBox(

                              height: 20,

                              width: 20,

                              child:
                                  CircularProgressIndicator(
                                strokeWidth: 2,
                              ),
                            )

                          : const Padding(

                              padding:
                                  EdgeInsets.symmetric(
                                vertical: 14,
                              ),

                              child:
                                  Text(

                                'Predict',

                                style:
                                    TextStyle(

                                  fontSize: 16,
                                ),
                              ),
                            ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ==========================================================
  // CLEANUP
  // ==========================================================

  @override
  void dispose() {

    // --------------------------------------------------------
    // Dispose local DINOv2
    // --------------------------------------------------------

    _dino.dispose();

    // --------------------------------------------------------
    // Dispose local EfficientNet
    // --------------------------------------------------------

    _efficientNet.dispose();

    // --------------------------------------------------------
    // API has nothing to dispose in Flutter
    // --------------------------------------------------------

    super.dispose();
  }
}