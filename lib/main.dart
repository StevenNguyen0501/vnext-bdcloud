import 'dart:async';
import 'package:flutter/material.dart';
import 'package:demo_urine/start.dart';

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false, // Tắt hiển thị chữ debug
    home: MainApp(),
  ));
}

class MainApp extends StatefulWidget {
  const MainApp({Key? key}) : super(key: key);
  @override
  _MainAppState createState() => _MainAppState();
}

class _MainAppState extends State<MainApp> {
  @override
  void initState() {
    super.initState();
    // Sau 5 giây, chuyển đến trang mới nếu không có tương tác
    Timer(Duration(seconds: 2), () {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => MyApp()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold( // Thay MaterialApp bằng Scaffold
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Align(
            alignment: Alignment.center,
            child: Image.asset(
              'shibachibi2.png', // Điều chỉnh tên tệp hình ảnh dựa trên tên thực tế của tệp của bạn
              height: 300,
            ),
          ),
          SizedBox(
            height: 50,
          ),
          Text(
            'Urine Test App',
            style: TextStyle(
              fontFamily: 'Times New Roman',
              fontSize: 50.0,
            ),
          ),
        ],
      ),
    );
  }
}
