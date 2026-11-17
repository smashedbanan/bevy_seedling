# Unreleased

# Changes

- Update `hrtf` to 0.10.0, whose HRIR resampling ratio is the right way round
- Remove `FftSize::slice_count`: a change of direction now crossfades across one slice, `hrtf`'s own interpolation
- Load an embedded subject's HRIR sphere once per sample rate and share it between nodes, so a node after the first builds in a tenth of the time
- Add `FftSize::power_of_two_fft`, which rounds the slice so the FFT length is a power of two at the device's rate: the same output, two to four times faster

# Fixes

- Fix `HrtfNode` leaving part of its output unwritten when the block size isn't a multiple of the FFT size
- Fix a reused `HrtfNode` starting at the previous sound's gain and direction
- Let the HRIR ring out after the input goes silent
- Fix the HRTF growing louder with the device's sample rate, 6.8 dB at 96 kHz
- Fix an exactly level or median-plane direction missing the HRIR sphere and keeping the last HRTF, or the front's on a node's first use

# 0.5.0

# Changes

- Update Firewheel to 0.12.0 and Bevy to 0.19

# 0.4.0

# Changes

- Update Firewheel to 0.10.0 and Bevy to 0.18

# 0.2.0

# Features

- Introduced `HrtfNode::coeff_update_factor`, which provides a bit
  more granular control over parameter smoothing.

# 0.1.1

## Fixes

- Ensure docs build with all features

# 0.1.0

This version is the first published to crates.io.
