from model.lipnet_res3d import lipnet_res3d
net,test_func = lipnet_res3d(grid.input_dim, grid.output_dim, weights=None) 
weight_path = './data/checkpoints/lipnet_res3d_weights_multiuser_with_auth_2017_08_07__08_20_49-55-1.55408811569-0.00957627118644.hdf5'

net.load_weights(weight_path) 
net.save_weights(weight_path) 
