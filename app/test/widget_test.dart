// Basic smoke test -- the counter-demo test from `flutter create` doesn't apply anymore
// since main.dart was replaced with the camera screen (which needs a real camera and
// permissions, not something to unit-test with a plain widget pump). This just checks the
// app builds a MaterialApp without throwing.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:hri_social_cues/main.dart';

void main() {
  testWidgets('HriApp builds a MaterialApp', (WidgetTester tester) async {
    await tester.pumpWidget(const HriApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
