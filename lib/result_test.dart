import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;
import 'package:image_gallery_saver/image_gallery_saver.dart';

class PageResult extends StatefulWidget {
  final http.Response response;

  const PageResult({Key? key, required this.response}) : super(key: key);

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<PageResult> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: const Text('Result'),
      ),
      backgroundColor: Colors.white,
      body: Container(
        color: Colors.white,
        child: SingleChildScrollView(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  height: 20,
                ),
                Container(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Container(
                        height: 500,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: Colors.green,
                            width: 5,
                          ),
                          borderRadius: BorderRadius.circular(30),
                        ),
                        
                        child: ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: _buildImage(),
                            ),
                      ),
                      SizedBox(height: 20),
                    ],
                  ),
                ),
                SizedBox(
                  height: 20,
                ),
                getResponse(),
                if (!(widget.response.statusCode == 200))
                  SizedBox(
                    height: 1000,
                  ),
                SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImage() {
    if (widget.response.statusCode == 200) {
      final jsonData = json.decode(widget.response.body);
      final textData1 = json.encode(jsonData);
      Map<String, dynamic> data = json.decode(textData1);
      List<int> imageBytes = base64.decode(data['base_64']);

      Uint8List uint8List = Uint8List.fromList(imageBytes);

      return Container(
        decoration: BoxDecoration(
          border: Border.all(
            color: Colors.white,
            width: 5,
          ),
          borderRadius: BorderRadius.circular(30),
        ),
        margin: const EdgeInsets.symmetric(vertical: 10),
        padding: const EdgeInsets.all(10),
        child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Image.memory(
               uint8List,
              ),
            ),
      );
    } else {
      return Text(
          'Không thể tải hình ảnh. Mã trạng thái: ${widget.response.statusCode}');
    }
  }

  Widget getResponse() {
    if (widget.response.statusCode == 200) {
      final jsonData = json.decode(widget.response.body);
      final textData1 = json.encode(jsonData);
      Map<String, dynamic> data = json.decode(textData1);
      return DataTable(
        dataTextStyle: TextStyle(fontWeight: FontWeight.bold, fontSize: 10),
        dividerThickness: 1.0,
        columns: [
          DataColumn(
              label: Text('Label',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15))),
          DataColumn(
              label: Text(
            'Value',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
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
            DataCell(Text('Gravity')),
            DataCell(Text(data['Gravity']))
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
