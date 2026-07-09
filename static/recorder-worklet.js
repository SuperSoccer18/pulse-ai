/**
 * recorder-worklet.js  —  AudioWorklet processor for Pulse AI
 * =============================================================
 * Runs in the dedicated AudioWorklet thread (not the main thread).
 *
 * Collects float32 mono PCM samples from the microphone and posts
 * them to the main thread in FRAME_SAMPLES-sized chunks.  The main
 * thread forwards each chunk over WebSocket to the server.
 *
 * Why a worklet instead of MediaRecorder?
 *   - Raw PCM is available immediately, no container overhead.
 *   - Sample-accurate framing: we control exactly how many samples
 *     per message, matching the server's CHUNK_SAMPLES window.
 *   - No WebM header / codec parsing on the server side.
 */

const FRAME_SAMPLES = 4800;  // 0.1 s at 48 kHz — small enough for low latency,
                              // large enough to amortise WebSocket round-trips.
                              // Server accumulates these into CHUNK_SAMPLES (9600)
                              // windows before encoding.

class PulseRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(FRAME_SAMPLES);
    this._offset = 0;
    this._active = true;

    // Main thread sends "stop" to flush and terminate cleanly.
    this.port.onmessage = (e) => {
      if (e.data === "stop") {
        this._active = false;
        // Flush remaining samples (< FRAME_SAMPLES) as a final partial frame.
        if (this._offset > 0) {
          this.port.postMessage(
            { type: "pcm", buffer: this._buffer.slice(0, this._offset).buffer },
            [this._buffer.slice(0, this._offset).buffer]
          );
          this._offset = 0;
        }
        this.port.postMessage({ type: "done" });
      }
    };
  }

  /**
   * process() is called by the audio engine for every 128-sample render quantum.
   * inputs[0][0] is the first channel of the first input (mono mic).
   */
  process(inputs) {
    if (!this._active) return false;   // returning false removes the processor

    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0];   // Float32Array of 128 samples

    let srcOffset = 0;
    while (srcOffset < samples.length) {
      const space  = FRAME_SAMPLES - this._offset;
      const copy   = Math.min(space, samples.length - srcOffset);

      this._buffer.set(samples.subarray(srcOffset, srcOffset + copy), this._offset);
      this._offset += copy;
      srcOffset    += copy;

      if (this._offset === FRAME_SAMPLES) {
        // Post a copy (transfer the buffer to avoid a data clone)
        const out = this._buffer.buffer.slice(0);
        this.port.postMessage({ type: "pcm", buffer: out }, [out]);

        // Allocate a fresh buffer for the next frame
        this._buffer = new Float32Array(FRAME_SAMPLES);
        this._offset = 0;
      }
    }

    return true;   // keep processor alive
  }
}

registerProcessor("pulse-recorder", PulseRecorderProcessor);
