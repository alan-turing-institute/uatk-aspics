import numpy as np

from aspics.buffers import Buffers
from aspics.params import Params


class Snapshot:
    """
    Thin wrapper around the .npz file format for saving/loading snapshots.
    This enables loading existing snapshots from file, or generating new snapshots full of random data or zeros.
    It also has a function for seeding initial infections in the population.
    Each snapshot consists of the data buffers used by OpenCL, as well as additional static data about the population
    which is not used in the runtime simulation but may be used for seeding infections at the snapshot stage.
    """

    def __init__(
        self,
        nplaces,
        npeople,
        nslots,
        time,
        area_codes,
        not_home_probs,
        lockdown_multipliers,
        buffers,
        name="cache",
    ):
        self.name = name
        self.nplaces = nplaces
        self.npeople = npeople
        self.nslots = nslots
        self.time = time
        self.area_codes = area_codes
        self.not_home_probs = not_home_probs
        self.lockdown_multipliers = lockdown_multipliers
        self.buffers = buffers

    def update_params(self, new_params):
        try:
            self.buffers.params[:] = new_params.asarray()
        except ValueError as e:
            print(
                f"Snapshot.py caused an exception '{str(e)}'. This can happen if the parameters in the model "
                f"have changed after a snapshot has been created. Try deleting the snapshot file "
                f"'data/config/snapshots/{self.name}.npz' and re-running the model."
            )
            raise e

    def seed_prngs(self, seed):
        """
        Recomputes the random states of the PRNGs passed to the kernels.
        The simulator runs deterministically for the same snapshot state, so calling this function gives new
        PRNG values to get enable stochastic results for different runs.
        """
        np.random.seed(seed)
        self.buffers.people_prngs[:] = np.random.randint(
            np.uint32((1 << 32) - 1), size=self.npeople * 4, dtype=np.uint32
        )

    @classmethod
    def load_full_snapshot(cls, path):
        """Creates a snapshot by reading the .npz file from the provided path."""
        with np.load(path, allow_pickle=True) as file_data:
            nplaces = file_data["nplaces"]
            npeople = file_data["npeople"]
            nslots = file_data["nslots"]
            time = file_data["time"]
            area_codes = file_data["area_codes"]
            not_home_probs = file_data["not_home_probs"]
            lockdown_multipliers = file_data["lockdown_multipliers"]
            buffers = Buffers(**{name: file_data[name] for name in Buffers._fields})
            return cls(
                nplaces,
                npeople,
                nslots,
                time,
                area_codes,
                not_home_probs,
                lockdown_multipliers,
                buffers,
            )

    def num_bytes(self):
        """Returns size in bytes of this snapshot."""
        total = 0
        for name in self.buffers._fields:
            total += getattr(self.buffers, name).nbytes
        return total
