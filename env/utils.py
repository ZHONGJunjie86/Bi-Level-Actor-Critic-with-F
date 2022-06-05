def information_share(states, agents):
    pass


   # po
def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: # 不放自己
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:  # agent 类型才会放速度
                other_vel.append(other.state.p_vel)
        print("agent.state.p_vel", [agent.state.p_vel] ,"agent.state.p_pos", [agent.state.p_pos] ,
                                 "entity_pos",entity_pos ,"other_pos", other_pos ,"other_vel", other_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)