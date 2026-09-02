# Speech & Audio

This suite measures local speech-to-text, speaker diarization, and text-to-speech. Dutch prompts, spoken audio, transcripts, and answers are intentionally allowed; technical metadata and statuses remain English.

STT reports WER/CER and real-time factor on practical Dutch fixtures and pinned FLEURS `nl_nl` samples. Diarization reports frame-based DER with optimal anonymous-speaker mapping. TTS reports generation performance and back-transcription intelligibility while retaining WAV files for blind review.

Raw results include references, transcripts, model/provider identity, timing, GPU telemetry, diarization segments, generated audio, and back-transcriptions.
