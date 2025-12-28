import os
import asyncio
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    db_path = '/tmp/langgraph_minimal_standalone.db'
    for ext in ['', '-wal', '-shm']:
        try:
            if os.path.exists(db_path + ext):
                os.remove(db_path + ext)
        except Exception:
            pass

    builder = StateGraph(int)
    builder.add_node('add_one', lambda x: x + 1)
    builder.set_entry_point('add_one')
    builder.set_finish_point('add_one')

    config = {'configurable': {'thread_id': 't_minimal_standalone'}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        g = builder.compile(checkpointer=saver)
        out = await g.ainvoke(1, config)

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver2:
        g2 = builder.compile(checkpointer=saver2)
        state = await g2.aget_state(config)

    print({'out': out, 'has_state': bool(state and state.values is not None)})

if __name__ == '__main__':
    asyncio.run(main())
