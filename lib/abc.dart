import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;
import 'package:image_gallery_saver/image_gallery_saver.dart';

class App extends StatefulWidget {
  final http.Response response;

  const App({Key? key, required this.response}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<App> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: const Text('Result'),
      ),
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            getResponse(),
            SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget getResponse() {
    if (widget.response.statusCode == 200) {
      final jsonData = json.decode(widget.response.body);
      final textData1 = json.encode(jsonData);
      Map<String, dynamic> data = json.decode(textData1);
      return DataTable(
        dataTextStyle: TextStyle(fontWeight: FontWeight.bold,fontSize: 10),
        dividerThickness: 1.0,
        columns: [
          DataColumn(
              label:
                  Text('Label', style: TextStyle(fontWeight: FontWeight.bold,fontSize: 15))),
          DataColumn(
              label: Text(
            'Value',
            style: TextStyle(fontWeight: FontWeight.bold,fontSize: 15),
          )),
        ],
        rows: [
          DataRow(cells: [
            DataCell(Text('Bilirubin')),
            DataCell(Text(data['Bilirubin']))
          ]),
          DataRow(
              cells: [DataCell(Text('Blood')), DataCell(Text(data['Blood']))]),
          DataRow(cells: [
            DataCell(Text('Glucose')),
            DataCell(Text(data['Glucose']))
          ]),
          DataRow(cells: [
            DataCell(Text('Ketone')),
            DataCell(Text(data['Ketone']))
          ]),
          DataRow(cells: [
            DataCell(Text('Leukocytes')),
            DataCell(Text(data['Leukocytes']))
          ]),
          DataRow(cells: [
            DataCell(Text('Nitrite')),
            DataCell(Text(data['Nitrite']))
          ]),
          DataRow(cells: [
            DataCell(Text('Protein')),
            DataCell(Text(data['Protein']))
          ]),
          DataRow(cells: [
            DataCell(Text('Specific')),
            DataCell(Text(data['Specific']))
          ]),
          DataRow(cells: [
            DataCell(Text('Urobilinogen')),
            DataCell(Text(data['Urobilinogen']))
          ]),
          DataRow(cells: [DataCell(Text('pH')), DataCell(Text(data['pH']))]),
        ],
      ); // Return text data as a Text widget
    } else {
      return Text(
          'Không thể tải hình ảnh. Mã trạng thái: ${widget.response.statusCode}');
    }
  }
}
