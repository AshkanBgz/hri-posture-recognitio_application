import 'dart:math' as math;

List<double> softmax(List<double> logits) {
  final maxLogit = logits.reduce(math.max);
  final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
  final sum = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sum).toList();
}
