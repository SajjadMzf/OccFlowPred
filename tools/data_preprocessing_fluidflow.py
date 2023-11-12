import os 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt  
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
# from waymo_open_dataset.protos import sys
# from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
# from waymo_open_dataset.utils import occupancy_flow_renderer
# from waymo_open_dataset.utils import occupancy_flow_vis

from tqdm import tqdm
from PIL import Image as Image
# from time import time as time
# import time 
# from data_utils import road_label,road_line_map,light_label,light_state_map
# from grid_utils import create_all_grids,rotate_all_from_inputs,add_sdc_fields
import sys
sys.path.append('/home/wmg-5gcat/Desktop/ofp/OFMPNet')
from core.utils.data_utils import road_label,road_line_map,light_label,light_state_map
from core.utils.grid_utils import create_all_grids,rotate_all_from_inputs,add_sdc_fields
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import argparse
from lbm.pylbm import LBM
import networkx as Nx
from skimage.measure import block_reduce

mpl.use('Agg')

PLOT = False
PLOT2 = False
def extract_lines(xy, id, typ):
    line = [] # a list of points  
    lines = [] # a list of lines
    IDs = []
    ID = []
    pl = 0
    pl_inline = 0
    pl_id = -1*np.ones(xy.shape)
    length = xy.shape[0]
    for i, p in enumerate(xy):
        line.append(p)
        ID.append(i)
        next_id = id[i+1] if i < length-1 else id[i]
        current_id = id[i]
        pl_id[i,0] = pl
        pl_id[i,1] = pl_inline
        pl_inline += 1
        if next_id != current_id or i == length-1:
            if typ in [18, 19]:
                line.append(line[0])
            lines.append(line)
            IDs.append(ID)
            line = []
            ID = []
            pl += 1
            pl_inline = 0
    return lines, IDs, pl_id

class Processor(object):

    def __init__(self, area_size, max_actors,max_occu, radius,rasterisation_size=256,save_dir='.',ids_dir=''):
        # parameters
        self.img_size = rasterisation_size # size = pixels * pixels
        self.area_size = area_size # size = [vehicle, pedestrian, cyclist] meters * meters
        self.max_actors = max_actors
        self.max_occu = max_occu
        self.radius = radius
        self.save_dir = save_dir
        self.ids_dir = ids_dir

        self.get_config()

    def load_data(self, filename):
        self.filename = filename
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        self.dataset_length = len(list(dataset.as_numpy_iterator()))
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        self.datalist = dataset.batch(1)
        #self.datalist = list(dataset.as_numpy_iterator())
    
    def get_config(self):
        config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        config_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 256
        grid_width_cells: 256
        sdc_y_in_grid: 192
        sdc_x_in_grid: 128
        pixels_per_meter: 3.2 
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(config_text, config)
        self.config = config

        ogm_config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        oconfig_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 512
        grid_width_cells: 512
        sdc_y_in_grid: 320
        sdc_x_in_grid: 256
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(oconfig_text, ogm_config)
        self.ogm_config = ogm_config
        self.config_2 = ogm_config
        # self.config = ogm_config

    def read_data(self, parsed):
        
        map_traj,real_map_traj,map_valid,actor_traj,traj_mask,occu_mask,actor_valid = rotate_all_from_inputs(parsed, self.config)
        map_traj_2,real_map_traj_2,map_valid_2,actor_traj_2,traj_mask_2,occu_mask_2,actor_valid_2 = rotate_all_from_inputs(parsed, self.config_2) 
        
        # actor traj
        self.actor_traj = actor_traj[0].numpy()
        self.traj_mask = traj_mask[0,:].numpy()
        self.occu_mask = occu_mask[0,:].numpy()
        #[batch,actor_num,11,1]
        self.actor_valid = actor_valid[0,:,:,0].numpy()
        self.actor_type = parsed['state/type'][0].numpy()

        # road map
        roadgraph_xyz = map_traj[0].numpy()
        real_map_traj = real_map_traj[0].numpy()
        roadgraph_dir = parsed['roadgraph_samples/dir'][0].numpy()
        roadgraph_type = parsed['roadgraph_samples/type'][0].numpy()
        roadgraph_id = parsed['roadgraph_samples/id'][0].numpy()
        roadgraph_valid = map_valid[0,:,0].numpy()
        # print(roadgraph_valid)
        v_mask = np.where(roadgraph_valid)
        self.roadgraph_xyz = roadgraph_xyz[v_mask]
        self.roadgraph_dir = roadgraph_dir[v_mask]
        self.roadgraph_type = roadgraph_type[v_mask]
        self.roadgraph_real_traj = real_map_traj[v_mask]
        self.roadgraph_id = roadgraph_id[v_mask]

        ######################## additional _2
        # actor traj
        self.actor_traj_2 = actor_traj_2[0].numpy()
        self.traj_mask_2 = traj_mask_2[0,:].numpy()
        self.occu_mask_2 = occu_mask_2[0,:].numpy()
        #[batch,actor_num,11,1]
        self.actor_valid_2 = actor_valid_2[0,:,:,0].numpy()
        self.actor_type_2 = parsed['state/type'][0].numpy()

        # road map
        roadgraph_xyz_2 = map_traj_2[0].numpy()
        real_map_traj_2 = real_map_traj_2[0].numpy()
        roadgraph_dir_2 = parsed['roadgraph_samples/dir'][0].numpy()
        roadgraph_type_2 = parsed['roadgraph_samples/type'][0].numpy()
        roadgraph_id_2 = parsed['roadgraph_samples/id'][0].numpy()
        roadgraph_valid_2 = map_valid_2[0,:,0].numpy()
        # print(roadgraph_valid)
        v_mask_2 = np.where(roadgraph_valid_2)
        self.roadgraph_xyz_2 = roadgraph_xyz_2[v_mask_2]
        self.roadgraph_dir_2 = roadgraph_dir_2[v_mask_2]
        self.roadgraph_type_2 = roadgraph_type_2[v_mask_2]
        self.roadgraph_real_traj_2 = real_map_traj_2[v_mask_2]
        self.roadgraph_id_2 = roadgraph_id_2[v_mask_2]

        ########################

        # print(len(self.roadgraph_id))
        self.roadgraph_uid = np.unique(self.roadgraph_id)
        # print(len(self.roadgraph_uid))
        self.roadgraph_types = np.unique(self.roadgraph_type)
        # print(self.roadgraph_types)

        # traffic lights
        traffic_light_state = parsed['traffic_light_state/current/state'][0].numpy()
        traffic_light_x = parsed['traffic_light_state/current/x'][0].numpy()
        traffic_light_y = parsed['traffic_light_state/current/y'][0].numpy()
        traffic_light_valid = parsed['traffic_light_state/current/valid'].numpy()
        self.traffic_light_x = traffic_light_x[0, np.where(traffic_light_valid)[1]]
        self.traffic_light_y = traffic_light_y[0, np.where(traffic_light_valid)[1]]
        self.traffic_light_state = traffic_light_state[0, np.where(traffic_light_valid)[1]]
    
    def actor_traj_process(self):
        emb = np.eye(3)
        traj_m = np.where(self.traj_mask)
        valid_actor = self.actor_traj[traj_m]
        valid_mask = self.actor_valid[traj_m]
        valid_type = self.actor_type[traj_m]
        dist=[]
        curr_buf=[]

        for i in range(valid_actor.shape[0]):
            w = np.where(valid_mask[i])[0]
            if w.shape[0]==0:
                continue
            n = w[-1]
            last_pos = valid_actor[i,n,:]
            dist.append(last_pos[:2])

        dist = np.argsort(np.linalg.norm(dist,axis=-1))[:self.max_actors]
        # current_state = [curr_buf[d] for d in dist]
        actor_type = []
        for d in dist:
            ind = int(valid_type[d])
            if ind in set([1,2,3]):
                actor_type.append(emb[ind-1])
            else:
                actor_type.append([0,0,0])

        output_actors = np.zeros((self.max_actors,11,5+3))
        for i,d in enumerate(dist):
            output_actors[i] =  np.concatenate((valid_actor[d],np.tile(actor_type[i],(11,1))),axis=-1)
        
        #process the possible occulde traj:
        occ_m = np.where(self.occu_mask)
        occu_actor = self.actor_traj[occ_m]
        occu_valid = self.actor_valid[occ_m]
        occu_type = self.actor_type[occ_m]

        dist=[]
        curr_buf=[]
        occu_traj = []
        o_type = []
        for i in range(occu_actor.shape[0]):
            w = np.where(occu_valid[i])[0]
            if w.shape[0]==0:
                continue
            b,e = w[0] , w[-1]
            begin_pos,last_pos = occu_actor[i,b,:2],occu_actor[i,e,:2]
            begin_dist,last_dist = np.linalg.norm(begin_pos) , np.linalg.norm(last_pos)
            if begin_dist<=last_dist:
                continue
            dist.append(last_dist)
            # curr_buf.append(occu_actor[i,e,:])
            occu_traj.append(occu_actor[i])
            o_type.append(occu_type[i])
        
        dist = np.argsort(dist)[:self.max_occu]
        out_occu_type = []
        for d in dist:
            ind = int(o_type[d])
            if ind in set([1,2,3]):
                out_occu_type.append(emb[ind-1])
            else:
                out_occu_type.append([0,0,0])

        output_occu_actors = np.zeros((self.max_occu,11,5+3))
        for i,d in enumerate(dist):
            output_occu_actors[i] = np.concatenate((occu_traj[d] ,np.tile(out_occu_type[i],(11,1))),axis=-1)
        
        return output_actors , output_occu_actors #, np.array(current_state)
    
    def seg_traj(self,traj,emb,seg_length=10):
        # np = self.np
        traj = np.array(traj)
        traj_length = traj.shape[0]
        pad_length = seg_length - traj_length % seg_length
        embs = np.tile(emb,(traj_length,1))
        traj = np.concatenate((traj,embs),axis=-1)
        traj = np.concatenate((traj,np.zeros((pad_length,4+3))),axis=0).reshape((-1,seg_length,4+3))
        return traj

    def map_traj_process(self):
        ##segment all valid center traj in the ogm map##   
        # np = self.np 
        num_segs = 256   
        type_set = set(self.roadgraph_types)
        # self.centerlines = []
        seg_length = 10
        line_cnt = 0
        res_traj = []
        # emb = np.eye(3)
        if 1 in type_set or 2 in type_set or 3 in type_set or 18 in type_set:
            for uid in self.roadgraph_uid:
                mask = np.where(self.roadgraph_id==uid)[0]
                way_type = int(self.roadgraph_type[mask][0])
                if way_type not in set([1,2,3,18]):
                    continue
                if way_type in set([1,2]):
                    emb_type = [1,0,0]
                elif way_type==3:
                    emb_type = [0,1,0]
                else:
                    emb_type = [0,0,1]
                traj = self.roadgraph_real_traj[mask]
                seg_traj = self.seg_traj(traj,seg_length=seg_length,emb=emb_type)
                seg_traj_len = seg_traj.shape[0]

                line_cnt += seg_traj_len
                res_traj.append(seg_traj)
                if line_cnt>num_segs:
                    break
            res_traj = np.concatenate(res_traj,axis=0)[:num_segs]
            if res_traj.shape[0]<num_segs:
                res_traj = np.concatenate((res_traj, np.zeros((num_segs-res_traj.shape[0],10,4+3))),axis=0)
            return res_traj
        else:
            return np.zeros((num_segs,10,4+3))

    def ogm_process(self,inputs):
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs, self.ogm_config) # (inputs, self.ogm_config)
        gt_v_ogm = tf.concat([timestep_grids.vehicles.past_occupancy,timestep_grids.vehicles.current_occupancy],axis=-1)
        gt_o_ogm = tf.concat([tf.clip_by_value(
                timestep_grids.pedestrians.past_occupancy +
                timestep_grids.cyclists.past_occupancy, 0, 1),
            tf.clip_by_value(
                timestep_grids.pedestrians.current_occupancy +
                timestep_grids.cyclists.current_occupancy, 0, 1)
                ],axis=-1)
        ogm = tf.stack([gt_v_ogm,gt_o_ogm],axis=-1)
        return ogm[0].numpy().astype(np.bool_),timestep_grids

    def image_process(self,show_image=False,num=0):

        fig, ax = plt.subplots()
        dpi = 1
        size_inches = self.img_size / dpi

        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        fig.set_facecolor('k')
        ax.set_facecolor('k') #
        ax.grid(False)
        ax.margins(0)
        ax.axis('off')
        
        # plot static roadmap
        big=80
        for t in self.roadgraph_types:
            road_points = self.roadgraph_xyz[np.where(self.roadgraph_type==t)[0]]
            road_points = road_points[:, :2]
            point_id = self.roadgraph_id[np.where(self.roadgraph_type==t)[0]]
            if t in set([1, 2, 3]):
                lines,_,_  = extract_lines(road_points, point_id, t)
                for line in lines:
                    ax.plot([point[0] for point in line], [point[1] for point in line], 
                             color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big, alpha=1, zorder=1)
            elif t == 17: # plot stop signs
                ax.plot(road_points.T[0, :], road_points.T[1, :], road_line_map[t][1], color=road_line_map[t][0], markersize=road_line_map[t][2]*big)
            elif t in set([18, 19]): # plot crosswalk and speed bump
                rects,_,_  = extract_lines(road_points, point_id, t)
                for rect in rects:
                     area = plt.fill([point[0] for point in rect], [point[1] for point in rect], color=road_line_map[t][0], alpha=0.7, zorder=2)
            else: # plot other elements
                lines,_,_  = extract_lines(road_points, point_id, t)
                for line in lines:
                    ax.plot([point[0] for point in line], [point[1] for point in line], 
                            color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big)

        # plot traffic lights
        for lx, ly, ls in zip(self.traffic_light_x, self.traffic_light_y, self.traffic_light_state):
            light_circle = plt.Circle((lx, ly), 1.5*big, color=light_state_map[ls], zorder=2)
            ax.add_artist(light_circle)

        pixels_per_meter = 1#self.config.pixels_per_meter
        range_x = self.config.sdc_x_in_grid
        range_y = self.config.sdc_y_in_grid

        ax.axis([0,256,0,256])
        ax.set_aspect('equal')

        # convert plot to numpy array
        fig.canvas.draw()
        array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        array = array.reshape(fig.canvas.get_width_height() + (3,))[::-1,:,:]

        plt.close('all')

        # visualize the image                   
        if show_image:
            img = self.Image.fromarray(array, 'RGB')
            time.sleep(30)
            plt.imshow(array)
        return array

    def image_process_2(self,show_image=False,num=0, vec_flow=None):
        self.notValidFrm = []
        fig, ax = plt.subplots()
        # fig.subplots_adjust(0,0,1,1) # Reza
        dpi = 1
        size_inches = self.img_size / dpi

        fig.set_size_inches([self.config_2.grid_width_cells, self.config_2.grid_height_cells])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        fig.set_facecolor('k')
        ax.set_facecolor('k') #
        ax.grid(False)
        ax.margins(0)
        ax.axis('off')
        
        # plot static roadmap
        big=80
        pnts = self.roadgraph_xyz_2
        # ax.scatter(pnts[:,0], pnts[:,1], s = 1000, color = 'yellow')
        # drs = self.roadgraph_real_traj[:,2:] # self.roadgraph_dir
        # ax.quiver(pnts[:,0], pnts[:,1], -drs[:,0], drs[:,1], scale=50, color = 'red')   
        #################### LBM
        nx, ny = 256, 256 #int(self.config_2.grid_width_cells), int(self.config_2.grid_height_cells)
        S = LBM((1,nx,ny),nphase=1)
        def cb_vel(self): # -- velocity is (z,y,x,3)
            where = np.where(S.uField == 1)
            self.fields['u'][0,where[0],where[1],2]=S.ux[where]
            self.fields['u'][0,where[0],where[1],1]=S.uy[where]

            
            # self.fields['u'][0,1:-1,0,2]=.1  # specified vel from left
            # self.fields['u'][0,1:-1,-1,:]=self.fields['u'][0,1:-1,-2,:]  # "open" right      
        # create a convenience function for plotting velocity
        def myplot(self,prefix):
            vmag=((self.fields['u'][0]**2).sum(axis=-1))**0.5
            # vmag[np.where(self.fields['ns'][0,:,:,0])] = 'nan'
            VX, VY = self.fields['u'][0,:,:,2], self.fields['u'][0,:,:,1]
            angV = np.arctan2(VY,VX)
            toplot = angV
            toplot[np.where(self.fields['ns'][0,:,:,0])] = 'nan'
            figg, axx = plt.subplots(1,2)
            plt.subplots_adjust(left=.02, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)
            figg.set_size_inches(20, 11, forward = True)
            im0 = axx[0].imshow(toplot, cmap = 'hsv', vmax = np.pi, vmin = -np.pi, origin='lower' )
            im1 = axx[1].imshow(vmag, origin='lower' )
            # axx[0].scatter()
            axx[0].title.set_text('%s: Angle'%prefix);plt.colorbar(im0, ax=axx[0], shrink=.7)   
            axx[1].title.set_text('%s: Speed'%prefix);plt.colorbar(im1, ax=axx[1], shrink=.7)   
            plt.show()   
            figg.savefig('lbm_fig.png') 
        # use that in callbacks at different stages of the simulation
        def cb_postMacro(self):
            if(self.step%20==0):
                myplot(self,prefix='postMacro(%d)'%self.step)
                # figg = plt.figure();myplot(self,prefix='postMacro(%d)'%self.step);plt.show()     
                # figg.savefig('lbm_fig.png')
        # gather callbacks to pass to the simulation method
        if PLOT2:
            cb = {'postMacro':[cb_vel,cb_postMacro]} # cb={'postMacro':[cb_vel,cb_postMacro]}
        else:
            cb = {'postMacro':[cb_vel]}
    
        # specify/reassign field variables such as scattering (z,y,x,phase)
        S.ux = np.zeros((nx,ny))
        S.uy = np.zeros((nx,ny))
        S.uField = np.zeros((nx,ny))
        ##################     
        cnd = (np.where((self.roadgraph_type_2==1) | (self.roadgraph_type_2==2) | (self.roadgraph_type_2==3)))
        road_points = self.roadgraph_xyz_2[cnd[0]]
        road_points = road_points[:, :2]
        road_dirs = self.roadgraph_real_traj_2[cnd[0]]
        road_dirs = road_dirs[:,2:]
        point_id = self.roadgraph_id_2[cnd[0]]
        lines, IDs, pl_id = extract_lines(road_points, point_id, 1)
        #########################  road tree
        start = []
        end = []
        dist = np.array([])
        for iline, line in enumerate(lines):
            if len(line) >=2: 
                start.append(line[0])
                end.append(line[-1])
        start = np.array(start)
        end = np.array(end)
        centerLine_exist = True
        if start.size == 0: 
            centerLine_exist = False
            self.notValidFrm.append(self.ifrm)
        if centerLine_exist:
            xstart, xend = np.meshgrid(start[:,0],end[:,0])
            ystart, yend = np.meshgrid(start[:,1],end[:,1])
            dist = np.sqrt((xstart-xend)**2+(ystart-yend)**2)
        connected = np.where(dist<.1)
        g_edge = []
        G = Nx.DiGraph()
        G.add_nodes_from(range(len(lines)))
        for i_nt in range(connected[0].shape[0]):
            g_edge.append((connected[0][i_nt],connected[1][i_nt]))
        G.add_edges_from(g_edge)
        figG = plt.figure(figsize =(9, 9)) 
        Nx.draw_networkx(G, node_color ='green') 
        if PLOT: figG.savefig('graph.png')
        # valid actors
        traj_m = np.where(self.traj_mask_2)
        occu_m = np.where(self.occu_mask_2)
        mask_actor = traj_m; #traj_m or occu_m
        valid_actor = self.actor_traj_2[mask_actor]
        valid_mask = self.actor_valid_2[mask_actor]
        valid_type = self.actor_type_2[mask_actor]
        vec_actor = valid_actor[np.where(valid_type == 1)]

        #########################
        # iline = 0
        th = 10 # 0-->10
        points_x = vec_actor[:,th,0]
        points_y = vec_actor[:,th,1]
        pixels_per_meter = self.config_2.pixels_per_meter
        points_x = np.round(points_x * pixels_per_meter) + self.config_2.sdc_x_in_grid
        points_y = np.round(-points_y * pixels_per_meter) + self.config_2.sdc_y_in_grid
        poitns = np.column_stack((points_x, points_y))

        if PLOT:
            cmap = plt.get_cmap('jet')
            colors = [cmap(i) for i in np.linspace(0, 1, len(lines))]
            for iline, line in enumerate(lines):
                ax.plot([point[0] for point in line], [point[1] for point in line], 
                            color=road_line_map[1][0], linestyle=road_line_map[1][1], linewidth=road_line_map[1][2]*big, alpha=1, zorder=0)
        
            ax.quiver(points_x,points_y,vec_actor[:,th,2],-vec_actor[:,th,3], scale = 80, color='red', zorder=3)    
        ######################## finding line coresponding to actors
        cnodes = np.array([])
        # array_x, array_y = np.meshgrid(range(self.config_2.grid_width_cells),range(self.config_2.grid_height_cells))
        if centerLine_exist:
            for ivec in range(poitns.shape[0]):
                actor_dx = road_points - poitns[ivec,:]
                actor_d = np.sqrt(actor_dx[:,0]**2+actor_dx[:,1]**2)
                actor_v = vec_actor[ivec,th,2:4]*np.array([1,-1])
                actor_speed = np.sqrt(actor_v[0]**2+actor_v[1]**2) + .0001
                actor_v = actor_v/actor_speed
                actor_dotp = 1 - road_dirs[:,0]*actor_v[0] + road_dirs[:,1]*actor_v[1]
                min_idx = np.argmin(1*actor_d+2*actor_dotp)
                cnode = Nx.shortest_path(G,pl_id[min_idx,0]).keys()
                cnode = np.array(list(cnode))
                cnodes = np.append(cnodes,cnode)
                closest_point = lines[int(pl_id[min_idx,0])][int(pl_id[min_idx,1])]

                ##### channel for each actor (subsequent branch of centerlines)
                Nlines = 3
                active_lines0 = np.array([int(pl_id[min_idx,0])])# [int(pl_id[min_idx,0])]
                i_points0 = np.array([int(pl_id[min_idx,1])]) #[int(pl_id[min_idx,1])]
                ext = False
                ch_points = lines[active_lines0[0]][int(pl_id[min_idx,1])]
                ch_points = np.vstack((ch_points,ch_points))
                ch_dirs = road_dirs[int(IDs[active_lines0[0]][int(pl_id[min_idx,1])])]
                ch_dirs = np.vstack((ch_dirs,ch_dirs))
                Npnts_future = np.round(10*actor_speed)
                Npnts_future = np.minimum(Npnts_future, 100)
                while not ext:
                    active_lines = active_lines0.copy()
                    i_points = i_points0.copy()
                    ext = True
                    for nl in range(len(active_lines)): 
                        if (active_lines[nl] != -1): # skip finished lines
                            line_now = lines[active_lines[nl]]
                            i_now = i_points[nl] 
                            ext = False                    
                            if i_now+2 > len(line_now): # if reach end of the line
                                active_lines0[nl] = -1
                                # i_points0[nl] = -1
                                next_active_lines = np.array(list(G.successors(active_lines[nl])), dtype=int)
                                next_active_lines = next_active_lines[np.where(next_active_lines != active_lines[nl])] #[xn for xn in next_active_lines if xn != active_lines[nl]] # to avoid loop routes
                                active_lines0 = np.append(active_lines0, next_active_lines)  #active_lines0 + next_active_lines
                                i_points0 = np.append(i_points0,np.zeros(len(next_active_lines), dtype = int))  #i_points0 + [0]*len(next_active_lines)
                            else:
                                ch_points = np.vstack((ch_points,line_now[i_now+1]))
                                ch_dirs = np.vstack((ch_dirs,road_dirs[int(IDs[active_lines[nl]][i_now+1])]))
                                i_points0[nl] = i_points0[nl] + 1
                    if (ch_points.shape[0] > Npnts_future): ext = True

                ############# BC for lbm simulation
                ## Version 1
                # r_bc = 8 #5
                # pnt_start_bc = closest_point #closest_point # poitns[ivec,:]
                # mask_bc =  np.where((array_x-pnt_start_bc[0])**2+(array_y-pnt_start_bc[1])**2 < r_bc**2)
                # px, py = array_x[mask_bc], array_y[mask_bc]
                # S.ux[px,py] = -actor_v[0]*actor_speed*.003
                # S.uy[px,py] = -actor_v[1]*actor_speed*.003
                # S.uField[px,py] = 1       

                ## Version 2   
                # px, py = np.round(ch_points[:,0]).astype(int),  np.round(ch_points[:,1]).astype(int)
                # S.ux[px,py] = ch_dirs[:,0]*actor_speed*.0003
                # S.uy[px,py] = ch_dirs[:,1]*actor_speed*.0003
                # S.uField[px,py] = 1     
                #############   
                # ax.plot(ch_points[:,0], ch_points[:,1], 
                #     color='yellow', linestyle=road_line_map[1][1], linewidth=road_line_map[1][2]*big, alpha=1, zorder=1)
                # ax.plot(ch_points[:,0], ch_points[:,1], 
                #     color='yellow', linestyle=road_line_map[1][1], linewidth=600, alpha=1, zorder=1)
                sPlot = 8e5
                if PLOT: sPlot = 1e5
                # if PLOT:
                ax.scatter(ch_points[:,0], ch_points[:,1], ####################### 
                    color='yellow',s = sPlot, zorder=1)            
                #####
                if PLOT:
                    ax.quiver(points_x[ivec],points_y[ivec],vec_actor[ivec,th,2],-vec_actor[ivec,th,3], scale = 80, color='blue', zorder=4)
        ########################    
        ### BC for lbm simulation (all at once from occupancy flow)
        # vec_flow = np.rot90(vec_flow, 3)
        # vec_flow = np.fliplr(vec_flow)
        vec_flow = block_reduce(vec_flow, block_size=(2,2,1), func=np.mean) # np.resize(vec_flow, (nx,ny,2))
        mask_bc =  np.where((vec_flow[:,:,0]!=0.) | (vec_flow[:,:,1]!=0.))
        S.ux[mask_bc] = vec_flow[:,:,0][mask_bc]*.003
        S.uy[mask_bc] = vec_flow[:,:,1][mask_bc]*.003
        S.uField[mask_bc] = 1       
        # S.ux[:] = vec_flow[:,:,0]*.003
        # S.uy[:] = vec_flow[:,:,1]*.003
        # S.uField[:] = 1 
        
        ### 

        if PLOT:       
            for iline, line in enumerate(lines):
                clr = colors[iline]
                sc = 80
                if iline in cnodes:
                    clr = 'black'
                    sc = 60
                ax.quiver(road_points[IDs[iline],0], road_points[IDs[iline],1], road_dirs[IDs[iline],0], -road_dirs[IDs[iline],1], scale = sc,
                            color = clr, zorder=2)
            ax.scatter(start[:,0],start[:,1],color = 'blue', s = 20000)
            ax.scatter(end[:,0],end[:,1],color = 'black', s = 2000)

            # ax.scatter(road_points[IDs[iline],0], road_points[IDs[iline],1], s = 1000, color = 'black')  
            # extracting the velocity field for S.LBM

            # px, py = np.round(road_points[IDs[iline],0]).astype(int), np.round(road_points[IDs[iline],1]).astype(int)
            # S.ux[px,py] = -road_dirs[IDs[iline],0]*.03
            # S.uy[px,py] = road_dirs[IDs[iline],1]*.03
            # S.uField[px,py] = 1 

        # FLOW = self.FLOW
        # FLOW = np.rot90(FLOW)
        # FLOW = np.rot90(FLOW)
        # FLOW = np.fliplr(np.rot90(FLOW))

        ## FLOW = np.rot90(FLOW)
        # whr_flow = np.where(((FLOW[:,:,0] != 0) | (FLOW[:,:,0] != 0))) 
        # px, py = whr_flow[0], whr_flow[1]
        # S.ux[px,py] = S.ux[px,py] - FLOW[px,py,0]*.003
        # S.uy[px,py] = S.uy[px,py] - FLOW[px,py,1]*.003                    
        # S.uField[px,py] = 1      

        ax.axis([0,self.config_2.grid_width_cells,0,self.config_2.grid_height_cells])
        ax.set_aspect('equal')
        # plt.gca().set_position([0, 0, 1, 1])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        
        if PLOT: plt.show(); fig.savefig('myfig.pdf')
        # input()
        # convert plot to numpy array
        fig.canvas.draw()
        array_g = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        array_g = array_g.reshape(fig.canvas.get_width_height() + (3,))[::-1,:,:]
        array_g = np.dot(array_g[...,:3], [0.2989, 0.5870, 0.1140]).astype(bool).astype(int)
        array_g = block_reduce(array_g, block_size=(2,2), func=np.max) # np.resize(array_g, (nx,ny))
        plt.close('all')
        # plt.cla()
        S.fields['ns'][0,:,:,0] = 1-array_g
        self.simFields = S.sim(steps=101,callbacks=cb)
        # input()
        def afterSimPlot(self):
            road_line_map_2 = {1:['w', 'solid', 32], 2:['w', 'solid', 32], 3:['w', 'solid', 28], 6:['black', 'dashed', 2], 
                 7:['black', 'solid', 2], 8:['black', 'solid', 2], 9:['xkcd:yellow', 'dashed', 4], 10:['xkcd:yellow', 'dashed', 2], 
                 11:['xkcd:yellow', 'solid', 2], 12:['xkcd:yellow', 'solid', 3], 13:['xkcd:yellow', 'dotted', 1.5], 15:['y', 'solid', 4.5*1.5], 
                 16:['y', 'solid', 4.5*1.5], 17:['r', '-.', 40], 18:['b', 'solid', 3], 19:['xkcd:orange', 'solid', 13]}
            vmag=((self.simFields['u'][0]**2).sum(axis=-1))**0.5
            # vmag[np.where(self.fields['ns'][0,:,:,0])] = 'nan'
            VX, VY = self.simFields['u'][0,:,:,2], self.simFields['u'][0,:,:,1]  #### extracting velocity vector field
            angV = np.arctan2(VY,VX)
            toplot = angV
            toplot[np.where(self.simFields['ns'][0,:,:,0])] = 'nan'
            figg, axx = plt.subplots(1,2)
            plt.subplots_adjust(left=.02, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)
            figg.set_size_inches(20, 11, forward = True)
            im0 = axx[0].imshow(toplot, cmap = 'hsv', vmax = np.pi, vmin = -np.pi, origin='lower', zorder=1 )
            im1 = axx[1].imshow(vmag, origin='lower', zorder=1 )
            # axx[0].scatter()
            axx[0].title.set_text('Angle');plt.colorbar(im0, ax=axx[0], shrink=.7)   
            axx[1].title.set_text('Speed');plt.colorbar(im1, ax=axx[1], shrink=.7)   
            axx[0].set_facecolor('black')
            ###### ploting the rest on top of lbm results ######
            for t in self.roadgraph_types:
                road_points = self.roadgraph_xyz[np.where(self.roadgraph_type==t)[0]]
                road_points = road_points[:, :2]
                road_dirs = self.roadgraph_real_traj[np.where(self.roadgraph_type==t)[0]]
                road_dirs = road_dirs[:,2:]
                point_id = self.roadgraph_id[np.where(self.roadgraph_type==t)[0]]
                if t in set([1, 2, 3]): # [1, 2, 3]
                    lines,_,_ = extract_lines(road_points, point_id, t)
                    for line in lines:
                        axx[0].plot([point[0] for point in line], [point[1] for point in line], 
                                 color=road_line_map_2[t][0], linestyle=road_line_map_2[t][1], linewidth=road_line_map_2[t][2], alpha=1, zorder=0)
                        # pass
                # elif t == 17: # plot stop signs
                #     axx[0].plot(road_points.T[0, :], road_points.T[1, :], road_line_map[t][1], color=road_line_map[t][0], markersize=road_line_map[t][2]*big)
                # elif t in set([18, 19]): # plot crosswalk and speed bump
                #     rects, IDs, pl_id = extract_lines(road_points, point_id, t)
                #     for rect in rects:
                #          area = plt.fill([point[0] for point in rect], [point[1] for point in rect], color=road_line_map[t][0], alpha=0.7, zorder=2)
                else: # plot other elements
                    lines, IDs, pl_id = extract_lines(road_points, point_id, t)
                    for line in lines:
                        axx[0].plot([point[0] for point in line], [point[1] for point in line], 
                                color=road_line_map_2[t][0], linestyle=road_line_map_2[t][1], linewidth=road_line_map_2[t][2])

            # # plot traffic lights
            # for lx, ly, ls in zip(self.traffic_light_x, self.traffic_light_y, self.traffic_light_state):
            #     light_circle = plt.Circle((lx, ly), 1.5*big, color=light_state_map[ls], zorder=2)
            #     axx[0].add_artist(light_circle)

            axx[0].axis([0,self.config_2.grid_width_cells,0,self.config_2.grid_height_cells])
            axx[0].set_aspect('equal')
            #######


            plt.show()   
            figg.savefig('lbm_fig.png')             
        if PLOT2: afterSimPlot(self)
        ################################### imshow
        # fig2, ax2 = plt.subplots(figsize=(size_inches, size_inches))
        # # fig2.set_size_inches([size_inches, size_inches])
        # # fig2.set_dpi(dpi)

        # plt.imshow(array, origin='lower')
        # plt.gca().set_position([0, 0, 1, 1])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # ax2.set_xticklabels([])
        # ax2.set_yticklabels([])

        
        # if PLOT: plt.show(); fig2.savefig('myfig_array.png', dpi=dpi)
        
        flow_ff = np.flip(self.simFields['u'][0,:,:,1:3], axis=2)



        return flow_ff

    def gt_process(self,timestep_grids,flow_only=False):
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        if flow_only:
            gt_origin_flow = tf.concat(true_waypoints.vehicles.flow_origin_occupancy,axis=0).numpy()
            return gt_origin_flow
        gt_obs_ogm = tf.concat(true_waypoints.vehicles.observed_occupancy,axis=0).numpy().astype(np.bool_)
        gt_occ_ogm = tf.concat(true_waypoints.vehicles.occluded_occupancy,axis=0).numpy().astype(np.bool_)
        gt_flow = tf.concat(true_waypoints.vehicles.flow,axis=0).numpy()
        gt_origin_flow = tf.concat(true_waypoints.vehicles.flow_origin_occupancy,axis=0).numpy()
        return gt_obs_ogm,gt_occ_ogm,gt_flow,gt_origin_flow
    
    def get_ids(self,val=True):
        if val:
            path = f'{self.ids_dir}/validation_scenario_ids.txt'
        else:
            path = f'{self.ids_dir}/testing_scenario_ids.txt'
        with tf.io.gfile.GFile(path) as f:
            test_scenario_ids = f.readlines()
            test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
            self.test_scenario_ids = set(test_scenario_ids)
    
    def flow_process(self,timestep_grids):
        vec_hist_flow = timestep_grids.vehicles.all_flow[:,:,:,0,:]
        ped_byc_hist_flow = timestep_grids.pedestrians.all_flow[:,:,:,0,:] + timestep_grids.cyclists.all_flow[:,:,:,0,:]
        return vec_hist_flow[0].numpy(),ped_byc_hist_flow[0].numpy()
    
    # def build_saving_tfrecords(self,pred,val,num):
    #     if pred:
    #         self.get_ids(val=False)
    #         if not os.path.exists(f'{self.save_dir}/test/'):
    #             os.makedirs(f'{self.save_dir}/test/')
    #         writer = tf.io.TFRecordWriter(f'{self.save_dir}/test/'+f'{num}'+'new.tfrecords')
    #     if val:
    #         self.get_ids(val=True)
    #         if not os.path.exists(f'{self.save_dir}/val/'):
    #             os.makedirs(f'{self.save_dir}/val/')
    #         writer = tf.io.TFRecordWriter(f'{self.save_dir}/val/'+f'{num}'+'new.tfrecords')
        
    #     if not (pred or val):
    #         if not os.path.exists(f'{self.save_dir}/train/'):
    #             os.makedirs(f'{self.save_dir}/train/')
    #         writer = tf.io.TFRecordWriter(f'{self.save_dir}/train/'+f'{num}'+'new.tfrecords')
    #     return writer
    def build_saving_path(self,pred,val,shard,sc_id):
        if pred:
            if not os.path.exists(f'{self.save_dir}/test_numpy/'):
                os.makedirs(f'{self.save_dir}/test_numpy/')
            path = f"{self.save_dir}/test_numpy/{shard}_{sc_id}.npz"
            
        if val:
            if not os.path.exists(f'{self.save_dir}/val_numpy/'):
                os.makedirs(f'{self.save_dir}/val_numpy/')
            path = f"{self.save_dir}/val_numpy/{shard}_{sc_id}.npz"
            
    
        if not (pred or val):
            if not os.path.exists(f'{self.save_dir}/train_numpy/'):
                os.makedirs(f'{self.save_dir}/train_numpy/')
            path = f"{self.save_dir}/train_numpy/{shard}_{sc_id}.npz"
            
        return path
            
    def workflow(self,pred=False,val=False):
        self.ifrm = 0
        # self.pbar = tqdm(total=self.dataset_length)
        # num = self.filename.split('-')[3]
        # writer = self.build_saving_tfrecords(pred, val,num)
        self.pbar = tqdm(total=self.dataset_length)

        if pred:
            self.get_ids(val=False)
        elif val:
            self.get_ids(val=True)

        shard = self.filename.split('-')[-3]

        # T0 = time.time()
        for dataframe in self.datalist:
            # t0 = time.time()
            sc_id = dataframe['scenario/id'].numpy()[0]
            if self.ifrm >= 0: # self.ifrm == 110: # 136  110 ###############################################################################
                # if pred or val:
                #     sc_id = dataframe['scenario/id'].numpy()[0]
                #     if isinstance(sc_id, bytes):
                #         sc_id=str(sc_id, encoding = "utf-8") 
                #     if sc_id not in self.test_scenario_ids:
                #         self.pbar.update(1)
                #         continue
                if isinstance(sc_id, bytes):
                    sc_id=str(sc_id, encoding = "utf-8") 
                if (pred or val) and sc_id not in self.test_scenario_ids:
                    self.pbar.update(1)
                    continue

                dataframe = add_sdc_fields(dataframe)
                self.read_data(dataframe)

                ogm,timestep_grids = self.ogm_process(dataframe)

                output_actors,occu_actors = self.actor_traj_process()
                map_trajs = self.map_traj_process()

                vec_flow,byc_flow = self.flow_process(timestep_grids)
                self.FLOW = vec_flow
                image = self.image_process(show_image=False,num=self.ifrm)
                flow_ff = self.image_process_2(show_image=False,num=self.ifrm, vec_flow=vec_flow)
                if PLOT:
                    fig3, ax3 = plt.subplots(2,2)  
                    # VX, VY = self.simFields['u'][0,:,:,2], self.simFields['u'][0,:,:,1] 
                    VX, VY = flow_ff[:,:,0], flow_ff[:,:,1]
                    angV = np.arctan2(VY,VX) 
                    # angV[np.where((VX == 0.) & (VY == 0.) )] = 'nan'
                    ax3[1,1].imshow(angV, cmap = 'hsv', vmax = np.pi, vmin = -np.pi, origin='lower') 
                    # ax3[1,1].axis([0-rr,self.config_2.grid_height_cells+rr,0-rr,self.config_2.grid_height_cells+rr])  
                    ax3[1,1].axis([0,256,0,256])  
                    ax3[0,0].imshow(ogm[:,:,-1,0], origin='lower') 
                    VX2, VY2 = vec_flow[:,:,0], vec_flow[:,:,1]
                    angV2 = np.arctan2(VY2,VX2) 
                    ax3[0,1].imshow(angV2, cmap = 'hsv', vmax = np.pi, vmin = -np.pi, origin='lower')   
                    ax3[1,0].imshow(image[:,:,:], origin='lower')   
                    # ax3[1].imshow(ogm[batch_n,:,:,-1,0], origin='lower')  
                    plt.show()
                    fig3.savefig('all_plot.png')  

                # image = image.tobytes()
                # ogm = ogm.tobytes()

                # map_trajs = map_trajs.tobytes()
                # output_actors = output_actors.tobytes()
                # occu_actors = occu_actors.tobytes()

                # vec_flow,byc_flow = self.flow_process(timestep_grids)
                # vec_flow = vec_flow.tobytes()
                # byc_flow = byc_flow.tobytes()
                
                # feature = {
                #     'centerlines': tf.train.Feature(bytes_list=tf.train.BytesList(value=[map_trajs])),
                #     'actors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[output_actors])),
                #     'occl_actors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[occu_actors])),
                #     'ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ogm])),
                #     'map_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                #     'gt_obs_ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                #     'gt_occ_ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                #     'gt_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                #     'origin_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                #     'vec_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vec_flow])),
                #     'byc_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[byc_flow]))
                # }
                # if pred or val:
                #     feature['scenario/id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sc_id.encode('utf-8')]))
                # if not pred:
                #     gt_obs_ogm,gt_occ_ogm,gt_flow,gt_origin_flow=self.gt_process(timestep_grids)
                #     feature['gt_obs_ogm'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_obs_ogm.tobytes()]))
                #     feature['gt_occ_ogm'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_occ_ogm.tobytes()]))
                #     feature['gt_flow'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_flow.tobytes()]))
                #     feature['origin_flow'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_origin_flow.tobytes()]))

                # example = tf.train.Example(features=tf.train.Features(feature=feature))
                # writer.write(example.SerializeToString())

        
                feature = {
                    'centerlines': map_trajs,
                    'actors': output_actors,
                    'occl_actors': occu_actors,
                    'ogm': ogm,
                    'map_image': image,
                    'vec_flow': vec_flow,
                    'byc_flow': byc_flow,
                    'vec_fluidflow': flow_ff
                }

                if pred or val:
                    feature['scenario/id'] = sc_id.encode('utf-8')
                if not pred:
                    gt_obs_ogm,gt_occ_ogm,gt_flow,gt_origin_flow=self.gt_process(timestep_grids)
                    feature['gt_obs_ogm'] = gt_obs_ogm
                    feature['gt_occ_ogm'] = gt_occ_ogm
                    feature['gt_flow'] = gt_flow
                    feature['origin_flow'] = gt_origin_flow

                path = self.build_saving_path(pred, val, shard, sc_id)

                np.savez_compressed(path, **feature)                
                self.pbar.update(1)
            # print('\n','Frame: -------- ', self.ifrm, 'dTime [sec]: ------- ', time.time()-t0, 'Time [sec]: ------- ', time.time()-T0)
            self.ifrm += 1
            # if i>=64:
            #     break

        # writer.close()
        self.pbar.close()

# def process_training_data(filename):
#     print('Working on',filename)
#     processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
#     save_dir=args.save_dir, ids_dir=args.ids_dir)
#     processor.load_data(filename)
#     processor.workflow()
#     print(filename, 'done!')
def process_training_data(filename):
    print('Working on',filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow()
    print(filename, 'done!')

def process_val_data(filename):
    print('Working on', filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow(val=True)
    print(filename, 'done!')

def process_test_data(filename):
    print('Working on', filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow(pred=True)
    print(filename, 'done!')

if __name__=="__main__":
    from multiprocessing import Pool
    from glob import glob

    parser = argparse.ArgumentParser(description='Data-preprocessing')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="/media/wmg-5gcat/ssd-roger/Waymo_Dataset/occupancy_flow_challenge")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="/media/wmg-5gcat/ssd-roger/Waymo_Dataset/preprocessed_data2")
    parser.add_argument('--file_dir', type=str, help='Dataset directory',default="/media/wmg-5gcat/ssd-roger/Waymo_Dataset/tf_example")
    parser.add_argument('--pool', type=int, help='num of pooling multi-processes in preprocessing',default=32)
    args = parser.parse_args()

    # NUM_POOLS = args.pool

    # train_files = glob(f'{args.file_dir}/training/*')
    # print(f'Processing training data...{len(train_files)} found!')
    # print('Starting processing pooling...')
    # process_training_data(train_files[0])
    # with Pool(NUM_POOLS) as p:
    #     p.map(process_training_data, train_files[:1])

    # from multiprocessing import Pool
    # from glob import glob

    # parser = argparse.ArgumentParser(description='Data-preprocessing')
    # parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="./Waymo_Dataset/occupancy_flow_challenge")
    # parser.add_argument('--save_dir', type=str, help='saving directory',default="./Waymo_Dataset/preprocessed_data")
    # parser.add_argument('--file_dir', type=str, help='Dataset directory',default="./Waymo_Dataset/tf_example")
    # parser.add_argument('--pool', type=int, help='num of pooling multi-processes in preprocessing',default=1)
    # args = parser.parse_args()

    NUM_POOLS = args.pool

    train_files = glob(f'{args.file_dir}/training/*')
    print(f'Processing training data...{len(train_files)} found!')
    print('Starting processing pooling...')
    # process_training_data(train_files[0])
    with Pool(NUM_POOLS) as p:
        p.map(process_training_data, train_files[:30])
    
    val_files = glob(f'{args.file_dir}/validation/*')
    print(f'Processing validation data...{len(val_files)} found!')
    print('Starting processing pooling...')
    with Pool(NUM_POOLS) as p:
        p.map(process_val_data, val_files[:30])
    
    test_files = glob(f'{args.file_dir}/testing/*')
    print(f'Processing validation data...{len(test_files)} found!')
    print('Starting processing pooling...')
    with Pool(NUM_POOLS) as p:
        p.map(process_test_data, test_files[:30])
