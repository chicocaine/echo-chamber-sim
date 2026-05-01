import { useMemo } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

import type { GraphEdge, GraphNode } from '../lib/types'

interface NetworkGraphProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  liveOpinions?: number[]
  liveEdges?: GraphEdge[]
}

interface ForceNode extends GraphNode {
  fx?: number
  fy?: number
}

interface ForceLink extends GraphEdge {
  source: number
  target: number
}

function opinionColor(opinion: number): string {
  const clamped = Math.max(-1, Math.min(1, opinion))
  if (clamped < 0) {
    const t = clamped + 1
    const rb = Math.round(255 * t)
    return `rgb(255, ${rb}, ${rb})`
  }
  const t = 1 - clamped
  const gb = Math.round(255 * t)
  return `rgb(${gb}, ${gb}, 255)`
}

function fixedLayout(nodes: GraphNode[]): ForceNode[] {
  const count = Math.max(1, nodes.length)
  const radius = 150 + Math.sqrt(count) * 8

  return nodes.map((node, index) => {
    const angle = (index / count) * Math.PI * 2
    return {
      ...node,
      fx: Math.cos(angle) * radius,
      fy: Math.sin(angle) * radius,
    }
  })
}

export function NetworkGraph({ nodes, edges, liveOpinions, liveEdges }: NetworkGraphProps) {
  const isLive = liveOpinions !== undefined && liveOpinions.length > 0
  const displayEdges = isLive && liveEdges && liveEdges.length > 0 ? liveEdges : edges

  const graphData = useMemo(
    () => ({
      nodes: fixedLayout(nodes),
      links: displayEdges.map((edge) => ({ ...edge })),
    }),
    [nodes, displayEdges],
  )

  // Use live opinions for node coloring when streaming.
  const nodeColorFn = isLive
    ? (node: { id?: number }) => opinionColor(liveOpinions[node.id ?? 0] ?? 0)
    : (node: { opinion?: number }) => opinionColor(node.opinion ?? 0)

  if (nodes.length > 300) {
    return (
      <section className="panel">
        <header className="panel-header">
          <h2>Network Graph</h2>
          <p>Rendering disabled: live graph is capped at N=300 in MVP.</p>
        </header>
      </section>
    )
  }

  return (
    <section className="panel graph-panel">
      <header className="panel-header">
        <h2>Network Graph</h2>
        <p>Node color encodes opinion, size encodes activity rate.</p>
      </header>

      <div className="graph-wrap">
        <ForceGraph2D<ForceNode, ForceLink>
          graphData={graphData}
          width={560}
          height={360}
          cooldownTicks={0}
          nodeVal={(node) => node.activity_rate * 5 + 3}
          nodeColor={nodeColorFn}
          linkColor={() => 'rgba(15, 23, 42, 0.2)'}
          backgroundColor="rgba(255, 255, 255, 0)"
        />
      </div>
    </section>
  )
}
