export type OpinionMap = Record<number, number>
export type AgentTypeMap = Record<number, string>

// Module-level store so tick-driven opinion changes never touch React props
// and therefore never cause ForceGraph2D to re-initialise.
let opinions: OpinionMap = {}
let agentTypes: AgentTypeMap = {}

export function setTickOpinions(next: OpinionMap) {
  opinions = next
}

export function setAgentTypes(next: AgentTypeMap) {
  agentTypes = next
}

export function getOpinion(agentId: number): number {
  return opinions[agentId] ?? 0
}

export function getAgentType(agentId: number, fallback: string): string {
  return agentTypes[agentId] ?? fallback
}
