import 'dart:convert';
import 'dart:html';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;
import 'dart:js' as js;
import 'package:demo_urine/result_test.dart';

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? imageUrl;
  bool isFrameVisible = true;
  String response = '';

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
            isFrameVisible = false;
          });
        });

        reader.readAsDataUrl(file!);
      }
    });
  }

  void showAlert() {
    js.context.callMethod('alert', ['Please select photo']);
  }

  Future<void> sendImage() async {
    if (imageUrl == null) return;

    try {
      String originalString = imageUrl.toString();
      int commaIndex = originalString.indexOf(',');
      String resultString = originalString.substring(commaIndex + 1).trim();
      print(resultString); // In ra: "đoạn text sau dấu phẩy"

      final response = await http.post(
        Uri.parse('http://192.85.4.149:8000/process_image_Ciede20001'),
        body: jsonEncode(
          {
            "imageBase64": resultString,
            "petName": "THINHNPPP",
            "userName": "THINHABCsD",
            "userEmail": "THINHNPPP",
            "userPhoneNumber": "THINHNPPP"
          }
        ), // Sending image as base64
        headers: {'Content-Type': 'application/json'},
      );

      setState(() {
        // Handle response as needed
        this.response = response.body;
      });
      print(response);
      await Navigator.push(
          context, MaterialPageRoute(builder: (context) => PageResult(response:response)));
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Center(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Center(
                 child: Text('Urine App Test',style: TextStyle(fontFamily: 'Times New Roman',fontSize: 60),),

                ),
                if (isFrameVisible)
                  InkWell(
                    onTap: () {
                      print('Thực Hành Lập Trình Flutter');
                      pickImage();
                    },
                    child: Container(
                      color: Colors.white,
                      child: Padding(
                        padding: EdgeInsets.symmetric(horizontal: 15.0),
                        child: Container(
                          height: 500, // Điều chỉnh chiều cao tùy ý
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: Colors.greenAccent,
                              width: 5,
                            ),
                            borderRadius: BorderRadius.circular(30),
                          ),
                          margin: const EdgeInsets.symmetric(vertical: 30),
                          padding: const EdgeInsets.all(30),
                          child: Stack(
                            children: [
                              Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Image.asset(
                                      'assets/image.png', // Điều chỉnh tên tệp hình ảnh dựa trên tên thực tế của bạn
                                      height: 80,
                                    ),
                                    Text(
                                      "Select file",
                                      style: TextStyle(
                                        fontFamily: 'Times New Roman',
                                        color: Colors.green,
                                        fontSize: 20.0,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  )
                else
                  Container(
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: Colors.greenAccent,
                        width: 5,
                      ),
                      borderRadius: BorderRadius.circular(30),
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
                              height: 500, // Increased height
                              fit: BoxFit.contain,
                            ),
                          ),
                      ],
                    ),
                  ),
                SizedBox(
                  height: 10,
                ),
                Center(
                  child: Text(
                    'OR',
                    style: TextStyle(
                        color: Colors.black,
                        fontSize: 20,
                        fontFamily: "Times New Roman"),
                  ),
                ),
                SizedBox(
                  height: 10,
                ),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 15.0),
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor:
                          Colors.greenAccent, // Màu nền mới của button
                    ),
                    onPressed: () {
                      pickImage();
                    },
                    child: Center(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.camera_alt,
                            size: 30,
                            color: Colors.white,
                          ),
                          SizedBox(width: 10),
                          Text(
                            'Open Camera & Take Photo',
                            style: TextStyle(
                                fontFamily: 'Times New Roman',
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.white),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  height: 90,
                ),
                Container(
                  height: 50,
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 15.0),
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green, // Màu nền mới của button
                      ),
                      onPressed: () {
                        if (isFrameVisible)
                          showAlert();
                        else
                          sendImage();
                      },
                      child: Center(
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(width: 10),
                            Text(
                              'Continute',
                              style: TextStyle(
                                  fontFamily: 'Times New Roman',
                                  fontSize: 30,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                SizedBox(height: 20,),
              ],
            ),
          ),
        ),
      );
  }
}
