# imports
import torch
import os

class ShareModule():
    def __init__(self, group_name, world_size, rank):
        self.group_name = group_name
        self.world_size = world_size
        self.rank = rank
        self.tmp_path = "".join(x for x in self.group_name if x.isalnum())
    
    def share_obj(self, obj):# share from all GPUs to GPU 0
        if self.world_size <= 1:
            return [obj,]
        if self.rank==self.out_rank:
            torch.distributed.barrier()# rank==0 wait for GPUs to write data to disk
            obj = [obj, *[torch.load(r, map_location='cpu') for r in range(self.world_size) if not self.rank]]
            torch.distributed.barrier()# rank==0 wait for GPUs to write data to disk
        else:
            # write data to disk then wait for rank==0 to load
            torch.save(obj, self.tmp_path+self.rank)
            torch.distributed.barrier()# rank==0 wait for GPUs to write data to disk
            torch.distributed.barrier()# once out_rank GPU is done, delete the tmp files
            os.remove(self.tmp_path+self.rank)
        return obj
    
    def share_obj(self, obj):# share object from GPU 0 to all GPUs
        if self.tmp_path is not None:
            raise Exception# object already shared/loaded!
        if rank==self.out_rank:
            tmp_path = self.group_name
            torch.save(obj, tmp_path)
            self.tmp_path = tmp_path
            torch.distributed.barrier()# data written, waiting on GPUs to load
            torch.distributed.barrier()# data loaded by other GPUs, will delete file and continue
            os.remove(self.tmp_path)
            return obj
        else:
            dict = torch.load(tmp, map_location='cpu')
            torch.distributed.barrier()# tell rank==0 that all GPUs have loaded data
            return dict

    def __enter__(self, out_rank=0):
        self.out_rank = out_rank
        if self.rank != out_rank:
            torch.distributed.barrier()# wait for data to be written to disk
    
    def __exit__(self, type, value, traceback):
        pass

# with ShareModule(world_size, rank) as sh:
#     obj = {'blah': 'blah_str'}
#     obj = sh.share_obj(obj)


# 1 - All GPUs except first will pause
# 2 - First GPU will process data and call share_obj to write data to disk
# 3 - All GPUs except first will load the saved data
# 4 - 