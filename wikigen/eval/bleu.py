from nltk.translate.bleu_score import (corpus_bleu,
                                       sentence_bleu,
                                       SmoothingFunction)




def bleu(gold_sequence_dict, generated_sequence_dict):
    #https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score

    smooth_fn = SmoothingFunction()

    refs = []
    hyps = []
    for index in gold_sequence_dict:
        if index in generated_sequence_dict:
            refs.append(gold_sequence_dict[index].split())
            hyps.append(generated_sequence_dict[index].split())

    #micro_average_precision = corpus_bleu(refs, hyps, smoothing_function=smooth_fn.method1)

    pairs = zip(refs, hyps)
    sentence_bleus = [sentence_bleu([ref], hyp, smoothing_function=smooth_fn.method1)
                      for ref, hyp in pairs]

    macro_average_precision = 1.0*sum(sentence_bleus)/len(sentence_bleus)

    return macro_average_precision

