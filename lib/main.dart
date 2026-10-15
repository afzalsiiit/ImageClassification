import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'clip_predict.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'CLIP Zero-Shot Classification',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ClipPage(),
    );
  }
}

class ClipPage extends StatefulWidget {
  const ClipPage({super.key});

  @override
  State<ClipPage> createState() => _ClipPageState();
}

class _ClipPageState extends State<ClipPage> {
  final ImagePicker _picker = ImagePicker();
  final ClipClassifier _classifier = ClipClassifierImpl();

  File? _image;
  bool _loading = false;
  String _result = 'Select an image';
  String _confidence = '';
  String _inferenceTime = '';
  String _totalTime = '';

  Future<void> _pickImage(ImageSource source) async {
    final selected = await _picker.pickImage(
      source: source,
      maxWidth: 1280,
      maxHeight: 1280,
      imageQuality: 100,
    );
    if (selected == null || !mounted) return;

    setState(() {
      _image = File(selected.path);
      _result = 'Image ready';
      _confidence = '';
      _inferenceTime = '';
      _totalTime = '';
    });
  }

  Future<void> _predict() async {
    final image = _image;
    if (image == null) {
      setState(() => _result = 'Please select an image first');
      return;
    }

    setState(() {
      _loading = true;
      _result = 'Running CLIP...';
      _confidence = '';
      _inferenceTime = '';
      _totalTime = '';
    });

    try {
      final result = await _classifier.predict(image.path);
      if (!mounted) return;
      setState(() {
        _result = result['label'] as String;
        _confidence =
            '${((result['confidence'] as num) * 100).toStringAsFixed(2)}%';
        _inferenceTime =
            '${(result['inference_time_ms'] as num).toStringAsFixed(2)} ms';
        _totalTime =
            '${(result['total_time_ms'] as num).toStringAsFixed(2)} ms';
      });
    } catch (error) {
      if (mounted) {
        setState(() {
          _result = 'Prediction failed';
          _confidence = error.toString();
        });
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CLIP Zero-Shot Classification'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.grey.shade300),
                    color: Colors.grey.shade100,
                  ),
                  child: _image == null
                      ? const Center(
                          child: Icon(
                            Icons.image,
                            size: 80,
                            color: Colors.grey,
                          ),
                        )
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.file(_image!, fit: BoxFit.contain),
                        ),
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Model: OpenAI CLIP ViT-B/32',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                _result,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (_confidence.isNotEmpty) Text('Confidence: $_confidence'),
              if (_inferenceTime.isNotEmpty) Text('Inference: $_inferenceTime'),
              if (_totalTime.isNotEmpty) Text('Total: $_totalTime'),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _loading
                          ? null
                          : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _loading
                          ? null
                          : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo),
                      label: const Text('Gallery'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _loading ? null : _predict,
                  child: _loading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Padding(
                          padding: EdgeInsets.symmetric(vertical: 14),
                          child: Text('Predict'),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }
}
