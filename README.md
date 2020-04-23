# Encoding monophonic signal with variable wavelength

This repo contains: 
- Tooling to split monophonic signal into a set of waveforms and to reconstruct the signal
- Basic waveform autoencoder

Take a look at `demo-last-crossing-point.ipynb` and `demo-stretch.ipynb` for a short demo

Some examples: 
- [Voice, reconstructed from the latent vectors (stretch)](https://flakycdn.blob.core.windows.net/various/render-human-voice-stretch.wav)
- [Voice, reconstructed from the latent vectors (last crossing point)](https://flakycdn.blob.core.windows.net/various/render-human-voice.wav)
- [Voice, reconstructed from the original set of waveforms with fixed pitch (stretch)](https://flakycdn.blob.core.windows.net/various/reconstructed-fixed-pitch.wav)
- [AKWF waveforms library, encoded with the human voice autoencoder](https://flakycdn.blob.core.windows.net/various/akwf.wav)
- [Voice with fixed pitch and some gaussian smoothing applied in the latent space](https://flakycdn.blob.core.windows.net/various/latent-space-filter.wav)


Two methods for wavelength encoding are provided:

- encoding wavelength into a separate channel

![Encoding F0](https://github.com/gnhdnb/waveform-encoder/raw/master/readme/encoding-pitch-stretch.png "Encoding F0")

- encoding by last zero-crossing point

![Encoding F0](https://github.com/gnhdnb/waveform-encoder/raw/master/readme/encoding-pitch.png "Encoding F0")
