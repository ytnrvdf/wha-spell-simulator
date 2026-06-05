export function createPlacementStore() {
  let placements = [];
  let nextId = 1;

  return {
    add({ kind, sourceId, baseStrokes, transform }) {
      const placement = {
        id: `p${nextId++}`,
        kind,
        sourceId,
        baseStrokes,
        transform: { ...transform }
      };
      placements = [...placements, placement];
      return placement;
    },

    update(id, patch) {
      placements = placements.map((placement) =>
        placement.id === id
          ? { ...placement, transform: { ...placement.transform, ...patch } }
          : placement
      );
      return placements.find((placement) => placement.id === id) ?? null;
    },

    remove(id) {
      const removed = placements.find((placement) => placement.id === id) ?? null;
      placements = placements.filter((placement) => placement.id !== id);
      return removed;
    },

    get(id) {
      return placements.find((placement) => placement.id === id) ?? null;
    },

    clear() {
      placements = [];
      nextId = 1;
    },

    getPlacements() {
      return placements;
    },

    count() {
      return placements.length;
    }
  };
}
