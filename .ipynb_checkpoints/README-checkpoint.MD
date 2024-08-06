---

# Automatic Speech Recognition and Intent Classification for Banking Industries

## Automatic Speech Recognition & Intent Classification

## Introduction

Speech recognition encompasses methodologies and technologies that enable the recognition and translation of spoken language into text by computers. It is also known as automatic speech recognition (ASR).

ASR helps humans interact with computers by allowing the machine to understand spoken language, including speech patterns, speaking styles, dialects, accents, and phrases.

## Problem

How to recognize sentences from multilingual audio recordings and understand the intent.

## Solution

Preprocess transcribed text for intent recognition in multiple languages, clean and normalize the text, and then use pre-trained NLP models such as wav2vec2 and Whisper for generating transcription and identifying the intent.

## Business Metrics

- **Customer Satisfaction**: Indicates how well the ASR system meets user expectations and needs.
- **Adoption Rate**: Shows the acceptance and popularity of the ASR system among its target audience.
- **Retention Rate**: High retention indicates user satisfaction and loyalty.

## Machine Learning Metrics

- **Word Error Rate (WER)**: The number of errors (insertions, deletions, and substitutions) divided by the total number of words spoken. WER is a primary measure of accuracy for ASR systems. Lower WER indicates better performance.

## Data Understanding & Data Preparation

The dataset used is sourced from the paper by Gerz et al. termed MInDS-14. The dataset contains multilingual and cross-lingual intent detection from spoken data. It includes 14 intents identified in a commercial e-banking system. Each intent is associated with spoken examples across 14 distinct language varieties.

### Dataset Structure

```python
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 8168)
})
```

### Sample Used

We intend to use 5 distinct languages: "zh-CN", "ru-RU", "fr-FR", "en-US", "de-DE".

The sample is split into:
- **Train**: 60%
- **Validation**: 20%
- **Testing**: 20%

## Data Modeling

### Intent Classification

| Model                              | Accuracy   |
|------------------------------------|------------|
| facebook/wav2vec2-base             | 0.078040   |
| facebook/wav2vec2-base [en-US, en-GB, en-AU] | 0.075967   |
| facebook/wav2vec2-base [en-US]     | 0.0921990   |
| LSTM [en-US, en-GB, en-AU]         | 0.880000   |
| LSTM All Languages                 | 0.880000   |

### Automatic Speech Recognition

| Model                          | WER        |
|--------------------------------|------------|
| Whisper tiny                   | 30.982818  |
| Whisper tiny with preprocessing| 37.773196  |
| Whisper small                  | 20.910364  |
| Whisper medium                 | 19.415205  |

### Models Uploaded to Hugging Face

In addition to the models evaluated, the following models have been uploaded to Hugging Face for broader usage and further fine-tuning:

1. [Prasetyow12/whisper-tiny-finetuned-multilingual](https://huggingface.co/Prasetyow12/whisper-tiny-finetuned-multilingual)
2. [andreanstev/whisper-small-multilingual](https://huggingface.co/andreanstev/whisper-small-multilingual)
3. [anggari/whisper-medium-multi](https://huggingface.co/anggari/whisper-medium-multi)

## Conclusion

- The best model for intent classification is **LSTM** with an accuracy of **0.88**.
- The best accuracy achieved by the **facebook/wav2vec2-base** model is **0.0921990**.
- For Automatic Speech Recognition, the best model is **Whisper medium**, which has a WER score of **19.415205**.

---