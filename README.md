# Encoding monophonic signal with pitch variation

![Encoding F0](https://github.com/gnhdnb/waveform-encoder/raw/master/readme/encoding-pitch.png "Encoding F0")

This repo contains: 
- Tooling to split monophonic signal into a set of fixed-size waveforms and to reconstruct the signal
- Basic waveform autoencoder

Take a look at `demo.ipnb` for a short demo

Some examples: 
- [Voice, reconstructed from the original set of waveforms](https://flakycdn.blob.core.windows.net/various/reconstructed-human-voice.wav)
- [Voice, reconstructed from the latent vectors](https://flakycdn.blob.core.windows.net/various/render-human-voice.wav)
- [AKWF waveforms library, encoded with the human voice autoencoder](https://flakycdn.blob.core.windows.net/various/akwf.wav)
- [Voice with fixed pitch and some gaussian smoothing applied in the latent space](https://flakycdn.blob.core.windows.net/various/latent-space-filter.wav)
