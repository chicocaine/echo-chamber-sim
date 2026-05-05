interface Props {
  tick: number
  maxTick: number
  isPlaying: boolean
  playbackSpeed: number
  onPlay: () => void
  onPause: () => void
  onStop: () => void
  onStepLeft: () => void
  onStepRight: () => void
  onSeek: (tick: number) => void
  onSpeedChange: (speed: number) => void
  disabled: boolean
}

const SPEEDS = [0.5, 1, 2, 4, 8]

export function PlaybackBar({
  tick,
  maxTick,
  isPlaying,
  playbackSpeed,
  onPlay,
  onPause,
  onStop,
  onStepLeft,
  onStepRight,
  onSeek,
  onSpeedChange,
  disabled,
}: Props) {
  if (maxTick <= 0) return null

  return (
    <div className="playback-bar">
      <button
        className="playback-btn"
        onClick={onStop}
        disabled={disabled}
        title="Stop / Reset"
      >
        ⏹
      </button>

      <button
        className="playback-btn"
        onClick={onStepLeft}
        disabled={disabled || tick <= 0}
        title="Step left"
      >
        ⏮
      </button>

      <button
        className={`playback-btn ${isPlaying ? 'active' : ''}`}
        onClick={isPlaying ? onPause : onPlay}
        disabled={disabled}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>

      <button
        className="playback-btn"
        onClick={onStepRight}
        disabled={disabled || tick >= maxTick}
        title="Step right"
      >
        ⏭
      </button>

      <div className="playback-seek">
        <input
          type="range"
          min={0}
          max={maxTick}
          value={tick}
          onChange={e => onSeek(parseInt(e.target.value, 10))}
          disabled={disabled}
        />
      </div>

      <span className="playback-tick">
        {tick} / {maxTick}
      </span>

      <div className="playback-speed">
        {SPEEDS.map(s => (
          <button
            key={s}
            className={`playback-speed-btn ${playbackSpeed === s ? 'active' : ''}`}
            onClick={() => onSpeedChange(s)}
            disabled={disabled}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  )
}
