import 'dart:convert';
import 'dart:html';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'abc.dart';

class MyWidget extends StatefulWidget {
  const MyWidget({Key? key}) : super(key: key);

  @override
  State<MyWidget> createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  String? imageUrl;
  bool isFrameVisible = false;
  String response = '';
  bool isLoading = true;
  bool isSend = false;

  Future<void> pickImage() async {
    final input = FileUploadInputElement()..accept = 'image/*';
    input.click();

    input.onChange.listen((event) {
      final files = input.files;
      if (files!.isNotEmpty) {
        final file = files[0];
        final reader = FileReader();

        reader.onLoad.listen((event) {
          setState(() {
            imageUrl = reader.result as String?;
            isFrameVisible = true;
          });
        });

        reader.readAsDataUrl(file!);
      }
    });
  }

  Future<void> sendImage() async {
    if (imageUrl == null) return;


    try {
      final response = await http.post(
        Uri.parse('http://0.0.0.0:8001/process_image_L2distance'),
        body: jsonEncode({'imageBase64': imageUrl}), // Sending image as base64
        headers: {'Content-Type': 'application/json'},
      );

      setState(() {
        // Handle response as needed
        this.response = response.body;

      });
      await Navigator.push(
          context,
          MaterialPageRoute(
              builder: (context) => App(
                response: response,
              )));
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: const Text(
          'Image Selection',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20.0,
          ),
        ),
        centerTitle: true,
      ),
      body: Container(
        color: Colors.lightBlueAccent,
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (isFrameVisible)
              Container(
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(10),
                ),
                margin: const EdgeInsets.symmetric(vertical: 10),
                padding: const EdgeInsets.all(10),
                child: Column(
                  children: [
                    if (imageUrl != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.network(
                          imageUrl!,
                          height: 400, // Increased height
                          fit: BoxFit.contain,
                        ),
                      ),
                  ],
                ),
              )
            else
              Container(
                height: 400, // Adjust the height accordingly
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(10),
                ),
                margin: const EdgeInsets.symmetric(vertical: 10),
                padding: const EdgeInsets.all(10),
                child: Center(
                  child: Text(
                    "No Image Selected",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                    ),
                  ),
                ),
              ),
            const SizedBox(height: 20),
            if (imageUrl == null)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: pickImage,
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10.0),
                        ),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            Icons.camera_alt,
                            size: 40,
                            color: Colors.lightBlueAccent,
                          ),
                          const SizedBox(width: 10),
                          Text(
                            "Select Image",
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            const SizedBox(height: 20),
            if (imageUrl != null)
              Padding(
                padding: EdgeInsets.only(top: 20),
                child: Center(
                  child: ElevatedButton(
                    onPressed:() async {
                      sendImage();

                      },
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                      backgroundColor: Color(0xff64abbf),
                    ),
                    child: const Text(
                      "Process Image",
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 20,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
