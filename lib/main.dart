import 'package:flutter/material.dart';
import 'package:pbl6/image.dart';
import 'package:pbl6/list_view.dart';

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner:true ,
    home: MainApp(),
  ));
}

class MainApp extends StatelessWidget {
  const MainApp({Key? key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Colors.lightBlueAccent,
        body: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                SizedBox(height: 60), // Adjusted space above the icon
                Align(
                  alignment: Alignment.topCenter,
                  child: Image.asset(
                    'icon.png', // Adjust the image file name based on your actual file name
                    height: 150,
                  ),
                ),
                SizedBox(height: 20), // Adjusted space between icon and text
                Align(
                  alignment: Alignment.center,
                  child: Text(
                    '\n\ Unire Test',
                    style: TextStyle(
                        fontSize: 42,
                        fontWeight: FontWeight.bold,
                        color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                ),
                Spacer(), // Takes up remaining space between text and button
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => MyWidget(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  ),
                  child: Text(
                    'Start',
                    style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: Colors.lightBlueAccent),
                  ),
                ),
                SizedBox(height: 10,),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => List_View_Result(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  ),
                  child: Text(
                    'history',
                    style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: Colors.lightBlueAccent),
                  ),
                ),
                SizedBox(height: 60), // Adjusted space below the button
              ],
            ),
          ),
        ),
      ),
    );
  }
}
