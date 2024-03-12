import 'dart:convert';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'list_view.dart';

class View_Info extends StatefulWidget {

  const View_Info({Key? key, }) : super(key: key);

  @override
  State<View_Info> createState() => _View_info_result();
}
class _View_info_result extends State<View_Info> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: const Text('Result'),
      ),
      backgroundColor: Colors.white,
      body: SingleChildScrollView( // Wrap your content with SingleChildScrollView
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(height: 20,),
              Image.asset(
                'icon.png', // Adjust the image file name based on your actual file name
                height: 150,
              ),
              SizedBox(height: 30),
              getResponse(),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
  Widget _buildImage() {

    return Image.asset(
      'icon.png', // Adjust the image file name based on your actual file name
      height: 150,
    );
  }

  Widget getResponse() {

      Map<String, dynamic> testData = {
        'Bilirubin': '0.8',
        'Blood': '+',
        'Glucose': '100',
        'Ketone': 'Negative',
        'Leukocytes': '15',
        'Nitrite': 'Negative',
        'Protein': '30',
        'Specific': '1.025',
        'Urobilinogen': '0.2',
        'pH': '6.5',
      };
      final textData1 = json.encode(testData);
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
    }
}
