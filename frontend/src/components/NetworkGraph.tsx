import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { GraphEdge, GraphNode } from '../lib/types'
import { getAgentType, getOpinion, setAgentTypes } from './networkGraphState'

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

function opinionColor(opinion: number): string {
  const t = (opinion + 1) / 2
  const r = Math.round(220 * t + 80 * (1 - t))
  const g = Math.round(60 * t + 80 * (1 - t))
  const b = Math.round(60 * t + 220 * (1 - t))
  return `rgb(${r},${g},${b})`
}

function agentLabel(type: string): string {
  const labels: Record<string, string> = {
    stubborn: 'Stubborn (FJ)',
    flexible: 'Flexible (DeGroot)',
    passive: 'Passive',
    zealot: 'Zealot',
    bot: 'Bot',
    hk: 'HK Bounded',
    contrarian: 'Contrarian',
    influencer: 'Influencer',
  }
  return labels[type] ?? type
}

export const NetworkGraph = memo(function NetworkGraph({ nodes, edges }: Props) {
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })

  // Track container size
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setSize({ width, height })
        }
      }
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Build graph data ONCE. Does not depend on opinions.
  const graphData = useMemo(() => {
    // Populate module-level agent type lookup
    const nextTypes: Record<number, string> = {}
    for (const n of nodes) {
      nextTypes[n.id] = n.agent_type
    }
    setAgentTypes(nextTypes)

    const nodeMap = new Map<number, GraphNode>()
    for (const n of nodes) nodeMap.set(n.id, n)

    const filteredEdges = edges.filter(
      e => nodeMap.has(e.source) && nodeMap.has(e.target)
    )

    const seen = new Set<string>()
    const uniqueEdges = filteredEdges.filter(e => {
      const key = [Math.min(e.source, e.target), Math.max(e.source, e.target)].join('-')
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })

    const displayNodes = nodes.slice(0, 800).map(n => ({
      id: n.id,
      agentType: n.agent_type,
    }))

    const nodeIds = new Set(displayNodes.map(n => n.id))
    const displayEdges = uniqueEdges.filter(
      e => nodeIds.has(e.source) && nodeIds.has(e.target)
    ).slice(0, 5000)

    return { nodes: displayNodes, links: displayEdges }
  }, [nodes, edges])

  const nodeCanvasObject = useCallback(
    (node: { id: number; agentType: string }, ctx: CanvasRenderingContext2D) => {
      const nodeSize = 1.8
      const opinion = getOpinion(node.id)
      const color = opinionColor(opinion)

      ctx.beginPath()
      ctx.arc(node.x ?? 0, node.y ?? 0, nodeSize, 0, 2 * Math.PI)
      ctx.fillStyle = color
      ctx.fill()

      if (node.agentType === 'bot') {
        ctx.beginPath()
        ctx.arc(node.x ?? 0, node.y ?? 0, nodeSize + 2, 0, 2 * Math.PI)
        ctx.strokeStyle = '#f0a060'
        ctx.lineWidth = 0.8
        ctx.stroke()
      }
    },
    []
  )

  const nodeLabel = useCallback(
    (node: { id: number; agentType: string }) => {
      const opinion = getOpinion(node.id)
      const type = getAgentType(node.id, node.agentType)
      return `${agentLabel(type)} · opinion ${opinion.toFixed(3)}`
    },
    []
  )

  const linkColor = useCallback(() => 'rgba(120,130,160,0.18)', [])
  const linkWidth = useCallback(() => 0.4, [])

  useEffect(() => {
    const g = graphRef.current
    if (g) {
      g.d3Force('charge')?.strength(-12)
      g.d3Force('link')?.distance(20)
      // Keep default center force to prevent drift — just weaken the pull
      const center = g.d3Force('center')
      if (center) center.strength(0.05)
    }
  }, [])

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeCanvasObject={nodeCanvasObject}
        nodeLabel={nodeLabel}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalParticles={0}
        nodeRelSize={1}
        backgroundColor="rgba(0,0,0,0)"
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        autoPauseRedraw={false}
        minZoom={0.3}
        maxZoom={8}
        width={size.width}
        height={size.height}
        onEngineStop={() => {
          graphRef.current?.zoomToFit(400, 40)
        }}
      />
    </div>
  )
})
