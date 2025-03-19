import torch
import numpy as np
class HeatmapGenerator:

    def __init__(self, resolution, num_keypoints, sigma = 2):
        self.resolution = resolution
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        x = torch.arange(resolution[0])
        y = torch.arange(resolution[1])
        self.grid_map = torch.meshgrid(x,y)


    def __call__(self, keypoints):
        b = len(keypoints)
        num_keypoints = self.num_keypoints
        sigma = self.sigma
        resolution = self.resolution
        grid_map_x = self.grid_map[0]
        grid_map_y = self.grid_map[1]
        grid_map_x = grid_map_x.unsqueeze(0).unsqueeze(0).expand(b, num_keypoints, -1, -1).float()
        grid_map_y = grid_map_y.unsqueeze(0).unsqueeze(0).expand(b, num_keypoints, -1, -1).float()

        #for i in range(b):
        #for j in range(num_keypoints):
        #if keypoints[i,j,2] > 0 and keypoints[i,j,0] > 0 and keypoints[i,j,0] < resolution[0] and keypoints[i,j,1] > 0 and keypoints[i,j,1] < resolution[1] :
        x_keypoints = keypoints[:, :, 1].unsqueeze(2).unsqueeze(3).expand(-1, -1, resolution[0], resolution[1]).float()
        y_keypoints = keypoints[:,:,0].unsqueeze(2).unsqueeze(3).expand(-1, -1, resolution[0], resolution[1]).float()
        x_norm = (grid_map_x - x_keypoints)**2
        y_norm = (grid_map_y - y_keypoints)**2
        heatmaps = torch.exp (-(x_norm+y_norm) /(2*sigma**2)).float()
        heatmaps[keypoints[:,:,2]<1,:,:] = 0
        heatmaps[keypoints[:, :, 0] < 0] = 0
        #heatmaps[keypoints[:, :, 0] > resolution[0]] = 0
        heatmaps[keypoints[:, :, 1] < 0] = 0
        #heatmaps[keypoints[:, :, 1] > resolution[1]] = 0

        # let's sum up the heatmaps for now
        heatmaps = torch.sum(heatmaps, dim=0)
        return heatmaps

class DenseposeMapGenerator:
    def __init__(self, resolution, number_parts=25):
        self.resolution = resolution
        self.number_parts = number_parts

    def __call__(self, densepose_xy, densepose_uv, resolution=None):
        return self.encode_regression(densepose_xy, densepose_uv, resolution)
        #return self.encode_hotvector(densepose_xy, densepose_uv, resolution)

    # the part map is encoded in a single map
    def encode_regression(self, densepose_xy, densepose_uv, resolution=None):
        num_parts = self.number_parts
        if resolution is None:
            resolution = self.resolution
        num_maps = 3  # num_parts + 2
        # densepose map consists of two channels for u,v and the rest is a hot vector encoding of the part mask
        densepose_map = np.zeros((len(densepose_xy), num_maps, resolution[0], resolution[1]))
        #        densepose_map[:, self.number_parts - 1 + 2] = 1 #background mask
        densepose_xy = densepose_xy.astype(int)
        for i in range(num_parts):
            c = i + 1
            current_part_indices = (densepose_uv[:, 0] == c) & (densepose_xy[:, 0] < resolution[0]) & (
                        densepose_xy[:, 1] < resolution[1]) & (densepose_xy[:, 0] >= 0) & (densepose_xy[:, 1] >= 0)
            if current_part_indices.sum() < 1:  # this part has no assigned pixels
                continue
            x_indices = densepose_xy[:, 1][
                current_part_indices]  # [(current_part_indices) & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)]
            if len(x_indices) < 1:
                continue
            y_indices = densepose_xy[:, 0][
                current_part_indices]  # [(current_part_indices) & (densepose_xy[:, 1] < resolution)& (densepose_xy[:, 1] < resolution)]
            densepose_map[:, 0, x_indices, y_indices] = densepose_uv[:, 1][
                current_part_indices]  # & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)] #assign u
            densepose_map[:, 1, x_indices, y_indices] = densepose_uv[:, 2][
                current_part_indices]  # & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)] # assign v
            densepose_map[:, 2, x_indices, y_indices] = c/25
            # mask_index = i + 2
            # assign the hot vector for correponding indices
            # densepose_map[:, mask_index, x_indices, y_indices] = 1
            # densepose_map[:, num_parts-1+2, x_indices, y_indices] = 0
        # print('densepose_map shape')
        return torch.from_numpy(densepose_map[0]).float()

    #one hot vector representation for the part mask
    def encode_hotvector(self, densepose_xy, densepose_uv, resolution=None):
        num_parts = self.number_parts
        if resolution is None:
            resolution = self.resolution
        num_maps = num_parts + 2
        #densepose map consists of two channels for u,v and the rest is a hot vector encoding of the part mask
        densepose_map = np.zeros((len(densepose_xy), num_maps, resolution[0], resolution[1]))
        densepose_map[:, self.number_parts - 1 + 2] = 1 #background mask
        densepose_xy = densepose_xy.astype(int)
        uv_fac = 1
        for i in range(num_parts-1):
            c = i + 1
            current_part_indices = (densepose_uv[:,0]==c) & (densepose_xy[:, 0] < resolution[0]) & (densepose_xy[:, 1] < resolution[1]) &  (densepose_xy[:, 0] >=0) & (densepose_xy[:, 1] >=0)
            if current_part_indices.sum() < 1: # this part has no assigned pixels
                continue
            x_indices = densepose_xy[:, 0][current_part_indices] #[(current_part_indices) & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)]
            if len(x_indices) < 1:
                continue
            y_indices = densepose_xy[:,1][current_part_indices] #[(current_part_indices) & (densepose_xy[:, 1] < resolution)& (densepose_xy[:, 1] < resolution)]
            densepose_map[:, 0, x_indices, y_indices] = densepose_uv[:,1][current_part_indices]*uv_fac# & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)] #assign u
            densepose_map[:, 1, x_indices, y_indices] = densepose_uv[:,2][current_part_indices]*uv_fac # & (densepose_xy[:, 0] < resolution)& (densepose_xy[:, 1] < resolution)] # assign v
            mask_index = i + 2
            #assign the hot vector for correponding indices
            densepose_map[:, mask_index, x_indices, y_indices] = 1
            densepose_map[:, num_parts-1+2, x_indices, y_indices] = 0
        #print('densepose_map shape')
        return torch.from_numpy(densepose_map[0]).float()


if __name__ == '__main__':
    num_keypoints = 2
    keypoints = torch.zeros(2, num_keypoints,3)
    hm_gen = HeatmapGenerator((3, 3), num_keypoints, 1)
    keypoints[1,0,0] = 3
    keypoints[1, 0, 1] = 2
    keypoints[1,0,2] = 2
    hm = hm_gen(keypoints)






