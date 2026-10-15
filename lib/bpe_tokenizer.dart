import 'dart:convert';
import 'dart:io';

class CustomBPETokenizer {
  CustomBPETokenizer(this.path, this.maxLength) {
    final tokenizer =
        jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    final model = Map<String, dynamic>.from(tokenizer['model'] as Map);
    vocab = Map<String, dynamic>.from(model['vocab'] as Map);
    unkToken = model['unk_token'] as String;
    endOfWordSuffix = model['end_of_word_suffix'] as String;

    final merges = model['merges'] as List<dynamic>;
    for (var rank = 0; rank < merges.length; rank++) {
      final merge = merges[rank];
      final parts = merge is String
          ? merge.split(' ')
          : (merge as List<dynamic>).map((part) => part.toString()).toList();
      if (parts.length != 2)
        throw FormatException('Invalid BPE merge at rank $rank');
      mergeRanks['${parts[0]}\t${parts[1]}'] = rank;
    }
  }

  final String path;
  final int maxLength;
  late final Map<String, dynamic> vocab;
  late final String unkToken;
  late final String endOfWordSuffix;
  final Map<String, int> mergeRanks = {};
  late final Map<int, String> byteEncoder = _bytesToUnicode();

  static Map<int, String> _bytesToUnicode() {
    final bytes = <int>[];
    for (var value = 33; value <= 126; value++) bytes.add(value);
    for (var value = 161; value <= 172; value++) bytes.add(value);
    for (var value = 174; value <= 255; value++) bytes.add(value);

    final chars = List<int>.from(bytes);
    var extra = 0;
    for (var value = 0; value < 256; value++) {
      if (!bytes.contains(value)) {
        bytes.add(value);
        chars.add(256 + extra++);
      }
    }

    return {
      for (var i = 0; i < bytes.length; i++)
        bytes[i]: String.fromCharCode(chars[i]),
    };
  }

  List<String> _splitText(String text) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').toLowerCase();
    final pattern = RegExp(
      r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\sA-Za-z0-9]+|[A-Za-z]+|[0-9]",
    );
    return pattern
        .allMatches(normalized)
        .map((match) => match.group(0)!)
        .toList();
  }

  List<String> _bpe(String text) {
    final symbols = text.split('');
    if (symbols.isEmpty) return symbols;
    symbols[symbols.length - 1] += endOfWordSuffix;

    while (symbols.length > 1) {
      var bestIndex = -1;
      var bestRank = 1 << 30;
      for (var i = 0; i < symbols.length - 1; i++) {
        final rank = mergeRanks['${symbols[i]}\t${symbols[i + 1]}'];
        if (rank != null && rank < bestRank) {
          bestIndex = i;
          bestRank = rank;
        }
      }
      if (bestIndex < 0) break;

      symbols[bestIndex] += symbols.removeAt(bestIndex + 1);
    }
    return symbols;
  }

  List<int> encode(String text) {
    final startId = (vocab['<|startoftext|>'] as num).toInt();
    final endId = (vocab['<|endoftext|>'] as num).toInt();
    final ids = <int>[startId];

    for (final piece in _splitText(text)) {
      final encoded = utf8
          .encode(piece)
          .map((byte) => byteEncoder[byte]!)
          .join();
      for (final token in _bpe(encoded)) {
        ids.add(((vocab[token] ?? vocab[unkToken]) as num).toInt());
      }
    }

    ids.add(endId);
    if (ids.length > maxLength) {
      ids.length = maxLength;
      ids[maxLength - 1] = endId;
    }
    while (ids.length < maxLength) ids.add(endId);
    return ids;
  }
}
