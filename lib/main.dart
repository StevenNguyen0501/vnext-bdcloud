import 'package:flutter/material.dart';
import 'package:flutter_line_liff/flutter_line_liff.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? _userName = 'Loading...';
  String? _userEmail = 'Loading...';
  String? _profileImageURL = '';

  @override
  void initState() {
    super.initState();
    initializeLineLiff();
  }

  Future<void> initializeLineLiff() async {
    try {
      // Initialize LINE LIFF SDK
      await FlutterLineLiff().init(
        config: Config(liffId: '{2004033343-DxP6oEe7}'),
        successCallback: () {},
        errorCallback: (error) {
          print('Error initializing LINE LIFF SDK: $error');
        },
      );

      // Wait for SDK to be ready
      await FlutterLineLiff().ready;

      // Get user's profile information
      final Profile profile = await FlutterLineLiff().profile;
      
      // Update UI with user's information
      setState(() {
        _userName = profile.displayName;
        _profileImageURL = profile.pictureUrl;
      });
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('LINE LIFF Example'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _profileImageURL != null
                  ? CircleAvatar(
                      backgroundImage: NetworkImage(_profileImageURL!),
                      radius: 50,
                    )
                  : CircularProgressIndicator(),
              SizedBox(height: 20),
              Text('Name: $_userName'),
              Text('Email: $_userEmail'),
            ],
          ),
        ),
      ),
    );
  }
}
