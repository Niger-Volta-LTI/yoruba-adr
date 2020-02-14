import glob
import os
import numpy as np


def score_call_WER(run_record, gt_record):
    """
        score_call_WER(hypothesis, reference)
                      (run_record, gt_record):
        compute WER, aka Levenshtein distance for words. Computes big matrix
        Big O(n*m) time and space complexity, n = len of reference, m = len of hypothesis
    """

    run_record = run_record.strip().split(' ')
    gt_record = gt_record.strip().split(' ')

    # Init data
    dist = np.zeros((len(gt_record) + 1) * (len(run_record) + 1), dtype=np.uint32)
    dist = dist.reshape((len(gt_record) + 1, len(run_record) + 1))

    for i in range(len(gt_record) + 1):
        for j in range(len(run_record) + 1):
            if i == 0:
                dist[0][j] = j
            elif j == 0:
                dist[i][0] = i

    # Compute Levenshtein distance
    for i in range(1, len(gt_record) + 1):
        for j in range(1, len(run_record) + 1):
            if gt_record[i - 1] == run_record[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                # # Add a zero weight, skip any pairs in the wer equiv list.
                # if (gt_record[i - 1], run_record[j - 1]) in wer_equiv_tuples:
                #     continue

                substitution = dist[i - 1][j - 1] + 1
                insertion = dist[i][j - 1] + 1
                deletion = dist[i - 1][j] + 1

                dist[i][j] = min(substitution, insertion, deletion)

    numer = dist[len(gt_record)][len(run_record)]
    denom = max(len(gt_record), len(run_record))

    return {'wer_numer': numer, 'wer_denom': denom, 'wer': numer / denom}


if __name__ == '__main__':


    # read in targets
    target_text = []
    with open("/Users/iroro/github/yoruba-adr/data/test/targets.txt", 'r') as target_file_handler:
        x = target_file_handler.read().splitlines()
        target_text += x
    print("Found " + str(len(target_text)) + " lines of REFERENCE text to score")

    os.chdir("/Users/iroro/github/yoruba-adr/data/test")
    for pred_file in glob.glob("pred.*.txt"):
        print("\n\nScoring: " + pred_file)

        numers, denoms = 0, 0
        with open(pred_file) as pred_file_handler:
            preds = pred_file_handler.read().splitlines()
            print("Found " + str(len(preds)) + " lines of PREDICTED text to score")

            assert len(target_text) == len(preds)
            for i in range(len(target_text)):
                scores = score_call_WER(preds[i], target_text[i])
                numers += scores['wer_numer']
                denoms += scores['wer_denom']

        print("WER: " + str(float(numers / denoms)))

