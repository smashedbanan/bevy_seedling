//! A head-related transfer function (HRTF) node for
//! [Firewheel](https://github.com/BillyDM/Firewheel),
//! powered by [Fyrox](https://docs.rs/hrtf/latest/hrtf/)'s
//! [IRCAM](http://recherche.ircam.fr/equipes/salles/listen/download.html)-based HRIR.
//!
//! HRTFs can provide far more convincing spatialization compared to
//! simpler techniques. They simulate the way our bodies filter sounds
//! based on where they're coming from, allowing you to distinguish up/down,
//! front/back, and the more typical left/right.
//!
//! This simulation is moderately expensive. You'll generally want to avoid more
//! than 32-64 HRTF emitters, especially on less powerful devices.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use firewheel::{
    channel_config::{ChannelConfig, NonZeroChannelCount},
    diff::{Diff, Patch},
    dsp::{coeff_update::CoeffUpdateFactor, distance_attenuation::DistanceAttenuatorStereoDsp},
    event::ProcEvents,
    node::{
        AudioNode, AudioNodeInfo, AudioNodeProcessor, NodeError, ProcBuffers, ProcExtra, ProcInfo,
        ProcessStatus,
    },
};
use glam::Vec3;
use hrtf::{HrirSphere, HrtfContext, HrtfProcessor};
use std::{
    io::Cursor,
    num::NonZeroU32,
    sync::{Mutex, PoisonError},
};

mod subjects;

pub use firewheel::dsp::distance_attenuation::{DistanceAttenuation, DistanceModel};
pub use subjects::{Subject, SubjectBytes};

/// Head-related transfer function (HRTF) node.
///
/// HRTFs can provide far more convincing spatialization
/// compared to simpler techniques. They simulate the way
/// our bodies filter sounds based on where they’re coming from,
/// allowing you to distinguish up/down, front/back,
/// and the more typical left/right.
///
/// This simulation is moderately expensive. You’ll generally
/// want to avoid more than 32-64 HRTF emitters, especially on
/// less powerful devices.
#[derive(Debug, Clone, Diff, Patch)]
#[cfg_attr(feature = "bevy", derive(bevy_ecs::component::Component))]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct HrtfNode {
    /// The positional offset from the listener to the emitter.
    pub offset: Vec3,

    /// The amount of muffling (lowpass) in the range `[20.0, 20_480.0]`,
    /// where `20_480.0` is no muffling and `20.0` is maximum muffling.
    ///
    /// This can be used to give the effect of a sound being played behind a wall
    /// or underwater.
    ///
    /// By default this is set to `20_480.0`.
    ///
    /// See <https://www.desmos.com/calculator/jxp8t9ero4> for an interactive graph of
    /// how these parameters affect the final lowpass cuttoff frequency.
    pub muffle_cutoff_hz: f32,

    /// Distance attenuation parameters.
    pub distance_attenuation: DistanceAttenuation,

    /// The time in seconds of the internal smoothing filter.
    ///
    /// By default this is set to `0.015` (15ms).
    pub smooth_seconds: f32,

    /// If the resutling gain (in raw amplitude, not decibels) is less than or equal
    /// to this value, the the gain will be clamped to `0` (silence).
    ///
    /// By default this is set to "0.0001" (-80 dB).
    pub min_gain: f32,

    /// An exponent representing the rate at which DSP coefficients are
    /// updated when parameters are being smoothed.
    ///
    /// Smaller values will produce less "stair-stepping" artifacts,
    /// but will also consume more CPU.
    ///
    /// The resulting number of frames (samples in a single channel of audio)
    /// that will elapse between each update is calculated as
    /// `2^coeff_update_factor`.
    ///
    /// By default this is set to `5`.
    pub coeff_update_factor: CoeffUpdateFactor,
}

impl Default for HrtfNode {
    fn default() -> Self {
        Self {
            offset: Vec3::ZERO,
            muffle_cutoff_hz: 20480.0,
            distance_attenuation: Default::default(),
            smooth_seconds: 0.015,
            min_gain: 0.0001,
            coeff_update_factor: CoeffUpdateFactor(5),
        }
    }
}

/// Configuration for [`HrtfNode`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy_ecs::component::Component))]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct HrtfConfig {
    /// The number of input channels.
    ///
    /// The inputs are downmixed to a mono signal
    /// before spatialization is applied.
    ///
    /// Defaults to [`NonZeroChannelCount::STEREO`].
    pub input_channels: NonZeroChannelCount,

    /// The head-related impulse-response sphere.
    ///
    /// The data for this sphere is captured from subjects. Short
    /// "impulses" are played from all angles and recorded at the
    /// ear canal. The resulting recordings capture how sounds are affected
    /// by the subject's torso, head, and ears.
    ///
    /// Defaults to `HrirSource::Embedded(Subject::Irc1040)`.
    pub hrir_sphere: HrirSource,

    /// The size of the FFT processing block, which can be
    /// tuned for performance.
    pub fft_size: FftSize,
}

impl Default for HrtfConfig {
    fn default() -> Self {
        Self {
            input_channels: NonZeroChannelCount::STEREO,
            hrir_sphere: Subject::Irc1040.into(),
            fft_size: FftSize::default(),
        }
    }
}

/// Describes the size of the FFT processing block.
///
/// Generally, you should try to make the audio's processing buffer size
/// a multiple of [`slice_len`][FftSize::slice_len] if possible.
/// A change of direction crossfades across one slice. A block shorter than a slice renders only
/// part of that fade: it carries on in the next block unless the target jumped 50 degrees or
/// more since the last render, which finishes it at once.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub struct FftSize {
    /// The size of each slice.
    ///
    /// Defaults to 128.
    pub slice_len: usize,

    /// Rounds [`slice_len`][FftSize::slice_len] to the nearest length whose FFT, the slice plus
    /// the HRIR less one, is a power of two at the device's rate. The output is the same; the FFT
    /// runs two to four times faster than at a length such as 48 kHz's 512 + 558 - 1 = 1069, a
    /// prime.
    ///
    /// Defaults to `false`.
    pub power_of_two_fft: bool,
}

impl Default for FftSize {
    fn default() -> Self {
        Self {
            slice_len: 128,
            power_of_two_fft: false,
        }
    }
}

impl FftSize {
    // the slice for an HRIR of `hrir_len` taps
    fn slice_for(&self, hrir_len: usize) -> usize {
        if !self.power_of_two_fft {
            return self.slice_len;
        }
        let history = hrir_len - 1;
        let fft = self.slice_len + history;
        let above = fft.next_power_of_two();
        let below = above / 2;
        let rounded = if below > history && fft - below < above - fft {
            below
        } else {
            above
        };
        rounded - history
    }
}

/// Provides a source for the HRIR sphere data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum HrirSource {
    /// Load data from the subjects embedded in the binary itself.
    Embedded(Subject),
    /// Load arbitrary data from an in-memory slice.
    InMemory(SubjectBytes),
}

impl HrirSource {
    fn get_sphere(&self, sample_rate: u32) -> Result<HrirSphere, hrtf::HrtfError> {
        match &self {
            HrirSource::Embedded(subject) => HrirSphere::new(Cursor::new(*subject), sample_rate),
            HrirSource::InMemory(subject) => {
                HrirSphere::new(Cursor::new(subject.clone()), sample_rate)
            }
        }
    }

    // The gain that keeps the HRTF as loud at `sample_rate` as at the subject's own rate: `hrtf`
    // resamples each HRIR as a signal, so its taps, and the filter's gain, grow with the ratio.
    // Call after a load of the same source succeeded, which read the header this reads.
    fn rate_gain(&self, sample_rate: u32) -> f32 {
        let bytes = match self {
            HrirSource::Embedded(subject) => subject.as_ref(),
            HrirSource::InMemory(subject) => subject.as_ref(),
        };
        let recorded = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        recorded as f32 / sample_rate as f32
    }
}

// Each embedded subject's sphere at each rate, loaded once: loading and resampling one is nearly all
// of a voice's build, and Firewheel builds the voices one after another, off the audio thread, while
// the thread that asked waits. Only embedded subjects are keyed; in-memory bytes load every time.
static SPHERES: Mutex<Vec<((Subject, u32), HrirSphere)>> = Mutex::new(Vec::new());

// the loads `shared_sphere` made, as (subject, rate)
#[cfg(test)]
static LOADS: Mutex<Vec<(Subject, u32)>> = Mutex::new(Vec::new());

impl HrirSource {
    // Takes a lock, so never on the audio thread: `FyroxHrtfProcessor::new` and `new_stream` both
    // run off it, never in `process`. A failed load is not kept.
    fn shared_sphere(&self, sample_rate: u32) -> Result<HrirSphere, hrtf::HrtfError> {
        let HrirSource::Embedded(subject) = self else {
            return self.get_sphere(sample_rate);
        };
        let key = (*subject, sample_rate);
        let mut spheres = SPHERES.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, sphere)) = spheres.iter().find(|(cached, _)| *cached == key) {
            return Ok(sphere.clone());
        }
        let sphere = self.get_sphere(sample_rate)?;
        #[cfg(test)]
        LOADS.lock().unwrap().push(key);
        spheres.push((key, sphere.clone()));
        Ok(sphere)
    }
}

impl From<Subject> for HrirSource {
    fn from(value: Subject) -> Self {
        Self::Embedded(value)
    }
}

impl From<SubjectBytes> for HrirSource {
    fn from(value: SubjectBytes) -> Self {
        Self::InMemory(value)
    }
}

/// A wrapper for [`hrtf::HrtfError`] that implements [`std::error::Error`].
#[derive(Debug)]
pub struct HrtfError(pub hrtf::HrtfError);

impl core::fmt::Display for HrtfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            hrtf::HrtfError::InvalidFileFormat => write!(f, "Failed to parse file as HRIR sphere"),
            hrtf::HrtfError::InvalidLength(len) => write!(f, "Invalid HRIR length `{len}`"),
            hrtf::HrtfError::IoError(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for HrtfError {}

impl AudioNode for HrtfNode {
    type Configuration = HrtfConfig;

    fn info(&self, config: &Self::Configuration) -> Result<AudioNodeInfo, NodeError> {
        Ok(AudioNodeInfo::new()
            .debug_name("hrtf node")
            .channel_config(ChannelConfig::new(config.input_channels.get(), 2)))
    }

    fn construct_processor(
        &self,
        config: &Self::Configuration,
        cx: firewheel::node::ConstructProcessorContext,
    ) -> Result<impl firewheel::node::AudioNodeProcessor, NodeError> {
        FyroxHrtfProcessor::new(self, config, cx.stream_info.sample_rate)
    }
}

// cos 50 degrees, so a jump of 50 or more finishes the fade at once. Up to it, carrying a partial fade
// on clicks at most 1.1 times as much as finishing it at once, probed off the y = 0 ring (where hrtf
// 0.10 misses directions) over directions, phases, tones and slices of 128 and 512 in half-slice
// blocks. 50 is the top of the range probed; wider tilts and 44.1 kHz first pass 1.1 at 60.
const CARRY_MIN_COS: f32 = 0.6428;

// Added to every direction `hrtf` samples: 0.10.0 misses the sphere on exactly level rays, the
// x = 0 meridian and the y = +-0.5 rings (15 of 360 level azimuths at 48 kHz), then keeps the last
// HRTF, or the first point's (the front) on a voice's first use. Rays 1e-6 off them never missed.
const NUDGE: Vec3 = Vec3::splat(1.0e-5);

fn ray(direction: Vec3) -> hrtf::Vec3 {
    let nudged = direction + NUDGE;
    hrtf::Vec3::new(nudged.x, nudged.y, nudged.z)
}

struct FyroxHrtfProcessor {
    renderer: HrtfProcessor,
    offset: Vec3,
    // the `offset` the last render used
    last_offset: Vec3,
    // `hrtf` crossfades from this direction to `offset` across each slice
    prev_direction: Vec3,
    attenuation: DistanceAttenuation,
    attenuation_processor: DistanceAttenuatorStereoDsp,
    // `HrirSource::rate_gain` at the stream's rate
    gain: f32,
    // `fft_size`'s slice for this stream's HRIR
    slice_len: usize,
    muffle_cutoff_hz: f32,
    min_gain: f32,
    slice: Vec<f32>,
    rendered: Vec<(f32, f32)>,
    // the last `hrir_len - 1` input samples, since `hrtf` would otherwise
    // keep the padding of a partial slice as history
    history: Vec<f32>,
    // `history` twice, one copy per sampling vector, as `hrtf` wants it
    hrtf_history: Vec<f32>,
    hrir_len: usize,
    silent_frames: usize,
    prev_input_settled: bool,
    sphere_source: HrirSource,
    fft_size: FftSize,
}

impl FyroxHrtfProcessor {
    fn new(
        node: &HrtfNode,
        config: &HrtfConfig,
        sample_rate: NonZeroU32,
    ) -> Result<Self, NodeError> {
        let sphere = config
            .hrir_sphere
            .shared_sphere(sample_rate.get())
            .map_err(HrtfError)?;

        let hrir_len = sphere.len();
        let slice_len = config.fft_size.slice_for(hrir_len);

        let offset = node.offset.normalize_or(Vec3::Y);
        let mut processor = FyroxHrtfProcessor {
            renderer: HrtfProcessor::new(sphere, slice_len),
            offset,
            last_offset: offset,
            prev_direction: offset,
            attenuation: node.distance_attenuation,
            attenuation_processor: DistanceAttenuatorStereoDsp::new(
                firewheel::param::smoother::SmootherConfig {
                    smooth_seconds: node.smooth_seconds,
                    ..Default::default()
                },
                sample_rate,
                node.coeff_update_factor,
            ),
            gain: config.hrir_sphere.rate_gain(sample_rate.get()),
            slice_len,
            muffle_cutoff_hz: node.muffle_cutoff_hz,
            min_gain: node.min_gain,
            slice: Vec::new(),
            rendered: Vec::new(),
            history: Vec::new(),
            hrtf_history: Vec::new(),
            hrir_len,
            silent_frames: 0,
            prev_input_settled: true,
            sphere_source: config.hrir_sphere.clone(),
            fft_size: config.fft_size.clone(),
        };
        processor.allocate();

        Ok(processor)
    }

    fn apply_patches(
        &mut self,
        patches: impl IntoIterator<Item = HrtfNodePatch>,
        sample_rate: NonZeroU32,
    ) {
        let mut updated = false;

        for patch in patches {
            updated = true;

            match patch {
                HrtfNodePatch::Offset(offset) => {
                    let distance = offset.length().max(0.01);

                    self.attenuation_processor.compute_values(
                        distance,
                        &self.attenuation,
                        self.muffle_cutoff_hz,
                        self.min_gain,
                    );

                    self.offset = offset.normalize_or(Vec3::Y);
                }
                HrtfNodePatch::MuffleCutoffHz(muffle) => {
                    self.muffle_cutoff_hz = muffle;
                }
                HrtfNodePatch::DistanceAttenuation(a) => {
                    self.attenuation.apply(a);
                }
                HrtfNodePatch::SmoothSeconds(s) => {
                    self.attenuation_processor
                        .set_smooth_seconds(s, sample_rate);
                }
                HrtfNodePatch::MinGain(g) => {
                    self.min_gain = g;
                }
                HrtfNodePatch::CoeffUpdateFactor(c) => {
                    self.attenuation_processor.set_coeff_update_factor(c);
                }
            }
        }

        // snap to the new gain between sounds, like `SpatialBasicNode`
        if updated && self.prev_input_settled {
            self.attenuation_processor.reset();
        }
    }

    // the HRIR has finished ringing out
    fn at_rest(&self) -> bool {
        self.silent_frames >= self.history.len()
    }

    fn allocate(&mut self) {
        let slice_len = self.slice_len;
        let history_len = self.hrir_len - 1;

        self.slice = vec![0.0; slice_len];
        self.rendered = vec![(0.0, 0.0); slice_len];
        self.history = vec![0.0; history_len];
        // `hrtf` reallocates this if the length is wrong
        self.hrtf_history = vec![0.0; 2 * history_len];

        self.clear();
    }

    // must not allocate, since this runs on the audio thread
    fn clear(&mut self) {
        self.history.fill(0.0);
        self.silent_frames = self.history.len();
        self.prev_input_settled = true;

        self.snap_direction();
        self.attenuation_processor.reset();
    }

    fn snap_direction(&mut self) {
        self.prev_direction = self.offset;
        self.last_offset = self.offset;
    }

    fn render(&mut self, frames: usize, left: &mut [f32], right: &mut [f32]) {
        self.slice[frames..].fill(0.0);
        self.rendered.fill((0.0, 0.0));
        let (first, second) = self.hrtf_history.split_at_mut(self.history.len());
        first.copy_from_slice(&self.history);
        second.copy_from_slice(&self.history);

        self.renderer.process_samples(HrtfContext {
            source: &self.slice,
            output: &mut self.rendered,
            new_sample_vector: ray(self.offset),
            prev_sample_vector: ray(self.prev_direction),
            new_gain: self.gain,
            prev_gain: self.gain,
            history: &mut self.hrtf_history,
        });
        // `hrtf` fades across the whole slice, so a partial one gets only part of the way. The target
        // jumping wider than the crossover since the last render finishes at once; so does a carried
        // gap past 90 degrees, so the lerp never nears the zero vector and every turn settles.
        self.prev_direction = if frames == self.slice_len
            || self.last_offset.dot(self.offset) < CARRY_MIN_COS
            || self.prev_direction.dot(self.offset) < 0.0
        {
            self.offset
        } else {
            // shortcut: on a device whose blocks are shorter than a slice, a continuous turn of
            // about 20 degrees a block or more can click at every block boundary, more than snapping
            // would (half-slice blocks, a 1.5 kHz tone), and a jump of CARRY_MIN_COS (50 degrees)
            // or more clicks once. If turns click on such a device, render whole slices through a
            // FIFO at one slice of latency.
            let reached = (frames - 1) as f32 / (self.slice_len - 1) as f32;
            self.prev_direction
                .lerp(self.offset, reached)
                .normalize_or(self.offset)
        };
        self.last_offset = self.offset;

        for ((left, right), &(rendered_left, rendered_right)) in
            left.iter_mut().zip(right.iter_mut()).zip(&self.rendered)
        {
            *left = rendered_left;
            *right = rendered_right;
        }

        let input = &self.slice[..frames];
        let history_len = self.history.len();
        if frames >= history_len {
            self.history.copy_from_slice(&input[frames - history_len..]);
        } else {
            self.history.copy_within(frames.., 0);
            self.history[history_len - frames..].copy_from_slice(input);
        }

        self.silent_frames = match input.iter().rposition(|sample| *sample != 0.0) {
            Some(last) => frames - 1 - last,
            None => self.silent_frames.saturating_add(frames),
        };
    }
}

impl AudioNodeProcessor for FyroxHrtfProcessor {
    fn events(&mut self, info: &ProcInfo, events: &mut ProcEvents, _extra: &mut ProcExtra) {
        self.apply_patches(events.drain_patches::<HrtfNode>(), info.sample_rate);
    }

    fn bypassed(&mut self, _bypassed: bool) {
        self.clear();
    }

    fn process(
        &mut self,
        proc_info: &ProcInfo,
        buffers: ProcBuffers,
        _: &mut ProcExtra,
    ) -> ProcessStatus {
        if proc_info
            .in_silence_mask
            .all_channels_silent(buffers.inputs.len())
        {
            self.prev_input_settled = true;
            self.attenuation_processor.reset();

            if self.at_rest() {
                return ProcessStatus::ClearAllOutputs;
            }
        } else {
            self.prev_input_settled = buffers.inputs_settled_at_zero();
        }

        if self.at_rest() {
            self.snap_direction();
        }

        let ProcBuffers { inputs, outputs } = buffers;
        let (left, rest) = outputs.split_first_mut().unwrap();
        let right = &mut *rest[0];

        // render each slice right away, zero-padding a partial one, so the
        // output is never delayed
        let frames = proc_info.frames;
        let slice_len = self.slice_len;
        for start in (0..frames).step_by(slice_len) {
            let end = (start + slice_len).min(frames);

            for (frame, sample) in (start..end).zip(self.slice.iter_mut()) {
                *sample =
                    inputs.iter().map(|channel| channel[frame]).sum::<f32>() / inputs.len() as f32;
            }

            self.render(end - start, &mut left[start..end], &mut right[start..end]);
        }

        let clear_outputs =
            self.attenuation_processor
                .process(frames, left, right, proc_info.sample_rate_recip);

        if clear_outputs {
            self.attenuation_processor.reset();
            ProcessStatus::ClearAllOutputs
        } else {
            ProcessStatus::OutputsModified
        }
    }

    fn new_stream(
        &mut self,
        stream_info: &firewheel::StreamInfo,
        _store: &mut firewheel::node::ProcStreamCtx,
    ) {
        if stream_info.prev_sample_rate != stream_info.sample_rate {
            let sample_rate = stream_info.sample_rate.get();

            let sphere = self
                .sphere_source
                .shared_sphere(sample_rate)
                .expect("HRIR data should be in a valid format");

            self.hrir_len = sphere.len();
            self.slice_len = self.fft_size.slice_for(self.hrir_len);
            self.gain = self.sphere_source.rate_gain(sample_rate);
            self.renderer = HrtfProcessor::new(sphere, self.slice_len);
            self.attenuation_processor
                .update_sample_rate(stream_info.sample_rate);

            self.allocate();
        } else {
            self.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn rate48() -> NonZeroU32 {
        NonZeroU32::new(48_000).unwrap()
    }

    #[test]
    fn the_sphere_keeps_its_length_at_each_device_rate() {
        let len = |rate| {
            HrirSource::from(Subject::Irc1040)
                .get_sphere(rate)
                .unwrap()
                .len()
        };
        // 512 taps at the subject's 44.1 kHz. hrtf 0.8.1 kept 512 at every rate, but resampled
        // the taps themselves at the inverted ratio.
        assert_eq!(len(44_100), 512);
        assert!((555..=559).contains(&len(48_000)), "{}", len(48_000));
        assert!((1_112..=1_117).contains(&len(96_000)), "{}", len(96_000));
    }

    fn loads_at(rate: u32) -> usize {
        let subject = Subject::Irc1040;
        let loads = LOADS.lock().unwrap();
        loads
            .iter()
            .filter(|&&load| load == (subject, rate))
            .count()
    }

    #[test]
    fn voices_at_one_rate_share_one_sphere() {
        // The lock is held across a load, so a rate is loaded once per process, whichever test
        // builds a node at it first.
        let rate = NonZeroU32::new(32_000).unwrap();
        let build = || FyroxHrtfProcessor::new(&HrtfNode::default(), &HrtfConfig::default(), rate);
        build().unwrap();
        build().unwrap();
        assert_eq!(loads_at(32_000), 1);
    }

    #[test]
    fn each_rate_gets_a_sphere_of_its_own_length() {
        // As `the_sphere_keeps_its_length_at_each_device_rate`, through the shared spheres, two
        // nodes at each rate. Each rate is loaded once per process, even if another test got there
        // first.
        let len = |rate: u32| {
            let rate = NonZeroU32::new(rate).unwrap();
            let node = HrtfNode::default();
            let p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate).unwrap();
            FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate).unwrap();
            p.hrir_len
        };
        assert_eq!(len(44_100), 512);
        assert!((555..=559).contains(&len(48_000)), "{}", len(48_000));
        assert!((1_112..=1_117).contains(&len(96_000)), "{}", len(96_000));
        assert_eq!(
            (loads_at(44_100), loads_at(48_000), loads_at(96_000)),
            (1, 1, 1)
        );
    }

    #[test]
    fn partial_slices_render_as_whole_ones() {
        // 63b1f13's first fix: the history is the real input, never a padded slice's zeros.
        // `render` takes `self.slice[..frames]` filled, as `process` fills it.
        let node = HrtfNode {
            offset: Vec3::new(1.0, 0.0, -1.0),
            ..Default::default()
        };
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.05).sin()).collect();
        let run = |block: usize| {
            let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
            let mut out = Vec::new();
            for chunk in input.chunks(block) {
                p.slice[..chunk.len()].copy_from_slice(chunk);
                let (mut l, mut r) = (vec![0.0; chunk.len()], vec![0.0; chunk.len()]);
                p.render(chunk.len(), &mut l, &mut r);
                out.extend(l.into_iter().zip(r));
            }
            out
        };
        for (i, (a, b)) in run(128).iter().zip(&run(100)).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4,
                "frame {i}: {a:?}, {b:?}"
            );
        }
    }

    #[test]
    fn a_reused_voice_starts_at_its_new_gain() {
        // 63b1f13's second fix: with the input settled, a new offset snaps the gain, no glide.
        // Default attenuation: `Inverse`, reference 5 m, so 20 m is 5 / (5 + 15) = 0.25.
        let node = HrtfNode {
            offset: Vec3::new(0.0, 0.0, -1.0),
            ..Default::default()
        };
        let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
        p.apply_patches(
            [HrtfNodePatch::Offset(Vec3::new(0.0, 0.0, -20.0))],
            rate48(),
        );
        assert!(!p.attenuation_processor.gain.is_smoothing());
        assert!((p.attenuation_processor.gain.target_value() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn rendering_never_reallocates_the_history() {
        let mut p = FyroxHrtfProcessor::new(&HrtfNode::default(), &HrtfConfig::default(), rate48())
            .unwrap();
        let before = p.hrtf_history.as_ptr(); // the buffer handed to `HrtfContext::history`
        p.slice.fill(0.5);
        let (mut l, mut r) = (vec![0.0; 128], vec![0.0; 128]);
        p.render(128, &mut l, &mut r);
        assert_eq!(before, p.hrtf_history.as_ptr());
    }

    #[test]
    fn a_turn_fades_from_the_last_direction_across_one_slice() {
        // Six slices of a tone, turning from `first` to `then` at the fifth; each slice's left ear.
        let input: Vec<f32> = (0..768).map(|i| (i as f32 * 0.05).sin()).collect();
        let run = |first: Vec3, then: Vec3| {
            let node = HrtfNode {
                offset: first,
                ..Default::default()
            };
            let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
            let (mut l, mut r) = (vec![0.0; 128], vec![0.0; 128]);
            let mut lefts = Vec::new();
            for (i, chunk) in input.chunks(128).enumerate() {
                if i == 4 {
                    p.offset = then;
                }
                p.slice.copy_from_slice(chunk);
                p.render(128, &mut l, &mut r);
                lefts.push(l.clone());
            }
            lefts
        };
        // The history holds input only, so a voice held at either direction gives each end of the fade.
        let turned = run(Vec3::NEG_X, Vec3::X);
        let (stayed, arrived) = (run(Vec3::NEG_X, Vec3::NEG_X), run(Vec3::X, Vec3::X));
        let differ =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() > 0.1;
        assert!(
            differ(&stayed[4][..8], &arrived[4][..8])
                && differ(&stayed[4][120..], &arrived[4][120..])
        );
        assert!(
            (turned[4][0] - stayed[4][0]).abs() < 1e-4,
            "starts at the last direction"
        );
        assert!(
            (turned[4][127] - arrived[4][127]).abs() < 1e-4,
            "ends at the new one"
        );
        // `render` moves the last direction on, so the next slice has nothing left to fade.
        for (i, (a, b)) in turned[5].iter().zip(&arrived[5]).enumerate() {
            assert!((a - b).abs() < 1e-4, "sixth slice, frame {i}: {a}, {b}");
        }
    }

    #[test]
    fn a_partial_slice_carries_on_unless_the_target_jumped_wide() {
        // Five degrees either side of `CARRY_MIN_COS`'s 50, one 64-of-128 render after the target
        // jumps: the first carries the fade on, short of the new direction; the second lands on it.
        let last_direction = |deg: f32| {
            let a = deg.to_radians();
            let then = Vec3::new(a.sin(), 0.0, -a.cos());
            let node = HrtfNode {
                offset: Vec3::NEG_Z,
                ..Default::default()
            };
            let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
            p.offset = then;
            p.slice[..64].fill(0.5);
            p.render(64, &mut [0.0; 64], &mut [0.0; 64]);
            (p.prev_direction, then)
        };
        let (carried, then) = last_direction(45.0);
        assert_ne!(carried, then);
        let (snapped, then) = last_direction(55.0);
        assert_eq!(snapped, then);
    }

    #[test]
    fn a_fast_continuous_turn_keeps_carrying_though_its_gap_is_wide() {
        // Thirty degrees of azimuth a block, about 29 true: the carried gap settles near 57 degrees,
        // past `CARRY_MIN_COS`'s 50, but the target never jumps that far, so no block finishes the
        // fade early.
        let ring = |deg: f32| {
            let (c, a) = ((1.0f32 - 0.26 * 0.26).sqrt(), deg.to_radians());
            Vec3::new(c * a.cos(), 0.26, c * a.sin())
        };
        let node = HrtfNode {
            offset: ring(0.0),
            ..Default::default()
        };
        let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
        for i in 1..30 {
            p.offset = ring(30.0 * i as f32);
            p.slice[..64].fill(0.5);
            p.render(64, &mut [0.0; 64], &mut [0.0; 64]);
            assert_ne!(p.prev_direction, p.offset, "block {i}");
        }
    }

    #[test]
    fn a_carried_gap_past_ninety_degrees_finishes_the_fade() {
        // The target has not jumped, but the carried gap is a reversal: a lerp would run through the
        // zero vector, so the fade finishes at once.
        let mut p = FyroxHrtfProcessor::new(&HrtfNode::default(), &HrtfConfig::default(), rate48())
            .unwrap();
        p.offset = Vec3::X;
        p.last_offset = Vec3::X;
        p.prev_direction = Vec3::NEG_X;
        p.slice[..64].fill(0.5);
        p.render(64, &mut [0.0; 64], &mut [0.0; 64]);
        assert_eq!(p.prev_direction, p.offset);
    }

    #[test]
    fn a_reversal_in_short_blocks_settles() {
        // An exact reversal has no direction to carry a half-fade on to: it must finish at once,
        // not fade about halfway every block for good.
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
        let run = |first: Vec3, then: Vec3| {
            let node = HrtfNode {
                offset: first,
                ..Default::default()
            };
            let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
            let mut out = Vec::new();
            for (i, chunk) in input.chunks(64).enumerate() {
                p.offset = if i < 8 { first } else { then };
                p.slice[..64].copy_from_slice(chunk);
                let (mut l, mut r) = (vec![0.0; 64], vec![0.0; 64]);
                p.render(64, &mut l, &mut r);
                out.extend(l);
            }
            out
        };
        // The turn block is 8 (frames 512..576); the blocks from 10 on have a settled history.
        let (turned, arrived) = (run(Vec3::NEG_X, Vec3::X), run(Vec3::X, Vec3::X));
        for (i, (a, b)) in turned.iter().zip(&arrived).enumerate().skip(640) {
            assert!((a - b).abs() < 1e-4, "frame {i}: {a}, {b}");
        }
    }

    #[test]
    fn a_narrow_turn_in_short_blocks_does_not_click_at_each_block() {
        // Ten degrees a block on a ring off the horizontal, in blocks of half a slice. Finishing
        // each half-fade at once clicks at every block boundary, up to 0.025 in the second
        // difference there; carrying on from where the fade got to stays under 0.005. The
        // largest second difference inside a block is 0.0013.
        let ring = |deg: f32| {
            let (c, a) = ((1.0f32 - 0.26 * 0.26).sqrt(), deg.to_radians());
            Vec3::new(c * a.cos(), 0.26, c * a.sin())
        };
        let input: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.05).sin()).collect();
        let node = HrtfNode {
            offset: ring(0.0),
            ..Default::default()
        };
        let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
        let mut out = Vec::new();
        for (i, chunk) in input.chunks(64).enumerate() {
            p.offset = ring(10.0 * (i.clamp(12, 28) - 12) as f32);
            p.slice[..64].copy_from_slice(chunk);
            let (mut l, mut r) = (vec![0.0; 64], vec![0.0; 64]);
            p.render(64, &mut l, &mut r);
            out.extend(l);
        }
        let click = (13..29)
            .map(|b| (out[b * 64 + 1] - 2.0 * out[b * 64] + out[b * 64 - 1]).abs())
            .fold(0.0f32, f32::max);
        assert!(click < 0.01, "{click}");
    }

    #[test]
    fn a_continuous_turn_in_short_blocks_keeps_carrying() {
        // Fifteen degrees a block on a ring off the horizontal, in blocks of half a slice. Carrying
        // on throughout clicks up to 0.03 in the second difference at a block boundary; finishing
        // the fade whenever the carried gap was wide clicks 0.11, and always finishing it 0.06.
        let ring = |deg: f32| {
            let (c, a) = ((1.0f32 - 0.26 * 0.26).sqrt(), deg.to_radians());
            Vec3::new(c * a.cos(), 0.26, c * a.sin())
        };
        let input: Vec<f32> = (0..2880).map(|i| (i as f32 * 0.05).sin()).collect();
        let node = HrtfNode {
            offset: ring(0.0),
            ..Default::default()
        };
        let mut p = FyroxHrtfProcessor::new(&node, &HrtfConfig::default(), rate48()).unwrap();
        let mut out = Vec::new();
        for (i, chunk) in input.chunks(64).enumerate() {
            p.offset = ring(15.0 * (i.clamp(12, 42) - 12) as f32);
            p.slice[..64].copy_from_slice(chunk);
            let (mut l, mut r) = (vec![0.0; 64], vec![0.0; 64]);
            p.render(64, &mut l, &mut r);
            out.extend(l);
        }
        let click = (13..43)
            .map(|b| (out[b * 64 + 1] - 2.0 * out[b * 64] + out[b * 64 - 1]).abs())
            .fold(0.0f32, f32::max);
        assert!(click < 0.05, "{click}");
    }

    // `input` through `p` in device blocks of `block`, sliced as `process` slices them
    fn play(p: &mut FyroxHrtfProcessor, input: &[f32], block: usize) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for chunk in input.chunks(block) {
            let (mut l, mut r) = (vec![0.0; chunk.len()], vec![0.0; chunk.len()]);
            for start in (0..chunk.len()).step_by(p.slice_len) {
                let end = (start + p.slice_len).min(chunk.len());
                p.slice[..end - start].copy_from_slice(&chunk[start..end]);
                p.render(end - start, &mut l[start..end], &mut r[start..end]);
            }
            out.extend(l.into_iter().zip(r));
        }
        out
    }

    fn rounded(slice_len: usize) -> FftSize {
        FftSize {
            slice_len,
            power_of_two_fft: true,
        }
    }

    #[test]
    fn a_power_of_two_fft_takes_the_nearest_slice_above_the_history() {
        // The HRIR's taps at 44.1, 48 and 96 kHz: 1023 rounds up, 1069 down, 1626 up, as 1024 is
        // shorter than the 1115 taps' history.
        assert_eq!(rounded(512).slice_for(512), 513);
        assert_eq!(rounded(512).slice_for(558), 467);
        assert_eq!(rounded(512).slice_for(1115), 934);
        assert_eq!(rounded(128).slice_for(558), 467);
        assert_eq!(rounded(513).slice_for(512), 513);
        assert_eq!(FftSize::default().slice_for(558), 128);
    }

    #[test]
    fn a_power_of_two_fft_changes_no_output() {
        // A held direction, in 1024-frame blocks: 512-sample slices against 467 at 48 kHz, every
        // third slice partial.
        let mut seed = 0x1234_5678_u32;
        let input: Vec<f32> = (0..1024 * 16)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let node = HrtfNode {
            offset: Vec3::new(0.3, 0.2, -0.9),
            ..Default::default()
        };
        let run = |fft_size: FftSize| {
            let config = HrtfConfig {
                fft_size,
                ..Default::default()
            };
            let mut p = FyroxHrtfProcessor::new(&node, &config, rate48()).unwrap();
            (p.slice_len, play(&mut p, &input, 1024))
        };
        let (exact_len, exact) = run(FftSize {
            slice_len: 512,
            power_of_two_fft: false,
        });
        let (pow2_len, pow2) = run(rounded(512));
        assert_eq!((exact_len, pow2_len), (512, 1024 - (len_at(48_000) - 1)));
        for (i, (a, b)) in exact.iter().zip(&pow2).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5,
                "frame {i}: {a:?}, {b:?}"
            );
        }
    }

    fn len_at(rate: u32) -> usize {
        HrirSource::from(Subject::Irc1040)
            .shared_sphere(rate)
            .unwrap()
            .len()
    }

    #[test]
    fn the_hrtf_is_as_loud_at_each_device_rate() {
        // A 1 kHz tone ahead, each ear's level against the input's. Unscaled, 96 kHz is 6.8 dB
        // louder than 44.1 (20 log10 of 96 / 44.1), as `hrtf` resamples the HRIR as a signal.
        let level = |rate: u32| {
            let node = HrtfNode {
                offset: Vec3::new(0.0, 0.01, -1.0),
                ..Default::default()
            };
            let config = HrtfConfig {
                fft_size: rounded(512),
                ..Default::default()
            };
            let mut p =
                FyroxHrtfProcessor::new(&node, &config, NonZeroU32::new(rate).unwrap()).unwrap();
            let tone: Vec<f32> = (0..rate as usize / 2)
                .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / rate as f32).sin())
                .collect();
            let out = play(&mut p, &tone, 1024);
            let settled = rate as usize / 10;
            let power =
                |samples: &mut dyn Iterator<Item = f32>| samples.map(|s| s * s).sum::<f32>();
            let input = power(&mut tone[settled..].iter().copied());
            let left = power(&mut out[settled..].iter().map(|s| s.0));
            let right = power(&mut out[settled..].iter().map(|s| s.1));
            [
                10.0 * (left / input).log10(),
                10.0 * (right / input).log10(),
            ]
        };
        let base = level(44_100);
        for rate in [48_000, 96_000] {
            for (ear, (at, base)) in level(rate).iter().zip(base).enumerate() {
                assert!(
                    (at - base).abs() < 0.25,
                    "{rate} Hz, ear {ear}: {at} dB, {base} dB"
                );
            }
        }
    }

    #[test]
    fn exactly_level_and_median_rays_find_their_hrtf() {
        // A voice's first slices at each direction against the same direction tilted 0.001 rad.
        // Unnudged, 15 of the 360 level azimuths miss at 48 kHz (30 degrees among them) and play
        // the front's HRTF.
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.13).sin()).collect();
        let config = HrtfConfig {
            fft_size: rounded(512),
            ..Default::default()
        };
        let run = |offset: Vec3| {
            let node = HrtfNode {
                offset,
                ..Default::default()
            };
            let mut p = FyroxHrtfProcessor::new(&node, &config, rate48()).unwrap();
            play(&mut p, &input, 1024)
        };
        let ring = |deg: f32, y: f32| {
            let (c, a) = ((1.0 - y * y).sqrt(), deg.to_radians());
            Vec3::new(c * a.sin(), y, -c * a.cos())
        };
        let meridian = |deg: f32| {
            let a = deg.to_radians();
            Vec3::new(0.0, a.sin(), -a.cos())
        };
        let level = (0..360).map(|deg| ring(deg as f32, 0.0));
        let rings = (0..360)
            .step_by(15)
            .flat_map(|deg| [ring(deg as f32, 0.5), ring(deg as f32, -0.5)]);
        let median = (-45..=225).step_by(15).map(|deg| meridian(deg as f32));
        for direction in level.chain(rings).chain(median) {
            let tilted = Quat::from_rotation_x(0.001) * direction;
            let diff = run(direction)
                .iter()
                .zip(&run(tilted))
                .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
                .fold(0.0f32, f32::max);
            assert!(diff < 0.01, "{direction}: {diff}");
        }
    }
}
