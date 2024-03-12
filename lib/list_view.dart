import 'dart:convert';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'info_result.dart';
import 'package:http/http.dart' as http;
import 'package:dio/dio.dart';


class List_View_Result extends StatefulWidget {
  const List_View_Result({Key? key}) : super(key: key);

  @override
  _List_View_ResultState createState() => _List_View_ResultState();
}

class _List_View_ResultState extends State<List_View_Result> {
  List<Station> stations = List.generate(24, (index) => Station(index + 1));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trang chủ'),
        backgroundColor: Colors.blue,
      ),
      body: ListView.builder(
        itemCount: stations.length,
        itemBuilder: (context, index) {
          Station item = stations[index];
          return Padding(
            padding: EdgeInsets.symmetric(vertical: 5.0, horizontal: 5.0),
            child: GestureDetector(
              onTap: ()  async {
                final dio = Dio();
                try {
                  Response response = await dio.post('http://0.0.0.0:8000/items/', queryParameters: {
                    // Any query parameters you want to include can be added here
                  }, data: {
                    // Your JSON data goes here
                    'key1': 'value1',
                    'key2': 'value2',
                  });
                  print(response.data); // This will print the JSON response received from the server
                } catch (e) {
                  print('Error fetching data: $e');
                }

                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => View_Info(),
                  ),
                );
              },
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(10.0),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.3),
                      spreadRadius: 2,
                      blurRadius: 5,
                      offset: Offset(0, 3),
                    ),
                  ],
                ),
                child: ListTile(

                  title: Container(
                    child: Text('Station ${item.id}',
                    style: TextStyle(fontSize: 18.0),),

                  ),
                  trailing: Icon(Icons.arrow_forward_ios),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class Station {
  int id;

  Station(this.id);
}
