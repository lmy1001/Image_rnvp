import torch
import torch.nn as nn
from networks.encoders import FeatureEncoder
from networks.decoders import LocalCondRNVPDecoder
from networks.resnet import resnet18
import io
import yaml
import visdom

class Conditional_RNVP_with_image_prior(nn.Module):
    def __init__(self, **kwargs):
        super(Conditional_RNVP_with_image_prior, self).__init__()

        self.mode = kwargs.get('usage_mode')
        self.deterministic = kwargs.get('deterministic')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')

        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')
        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_base_var = kwargs.get('p_decoder_base_var')

        self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                      self.p_latent_space_size, deterministic=False,
                                      mu_weight_std=0.001, mu_bias=0.0,
                                      logvar_weight_std=0.01, logvar_bias=0.0)

        self.pc_decoder = LocalCondRNVPDecoder(self.p_decoder_n_flows,
                                               self.p_decoder_n_features,
                                               self.g_latent_space_size,
                                               weight_std=0.01)

        self.image_encoder = resnet18(num_classes=self.g_latent_space_size)
        self.g_prior_n_layers = kwargs.get('g_prior_n_layers')
        self.g0_prior = FeatureEncoder(self.g_prior_n_layers, self.g_latent_space_size,
                                       self.g_latent_space_size, deterministic=False,
                                       mu_weight_std=0.0033, mu_bias=0.0,
                                       logvar_weight_std=0.033, logvar_bias=0.0)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, p_input, images, n_sampled_points=None):
        sampled_cloud_size = p_input.shape[2] if n_sampled_points is None else n_sampled_points

        output = {}
        if self.mode == 'training':
            ####add image features as condition to the RNVP
            img_features = self.image_encoder(images)       # n * 512
            output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)      # N * latent
            output['g_prior_samples'] = self.reparameterize(output['g_prior_mus'], output['g_prior_logvars'])

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'])
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]

            buf_p = self.pc_decoder(p_input, output['g_prior_samples'], mode='inverse')
            output['p_prior_samples'] = buf_p[0] + [p_input]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        elif self.mode == 'predicting':
            img_features = self.image_encoder(images)
            output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)      # N * latent
            output['g_prior_samples'] = self.reparameterize(output['g_prior_mus'], output['g_prior_logvars'])       #n* 512

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'])      # N * 3
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                images.shape[0], self.p_latent_space_size, sampled_cloud_size
            )]      # N * 3 * 2500
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                images.shape[0], self.p_latent_space_size, sampled_cloud_size
            )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]       # N * 3 * 2500

            buf_p = self.pc_decoder(output['p_prior_samples'][0], output['g_prior_samples'], mode='direct')
            output['p_prior_samples'] += buf_p[0]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        return output

class Model_of_Full_Obj(nn.Module):
    def __init__(self, num_seg_classes, **config):
        super(Model_of_Full_Obj, self).__init__()

        model_list = []
        for i in range(num_seg_classes):
            cur_model = Conditional_RNVP_with_image_prior(**config)
            model_list.append(cur_model)
            cur_model = None

        self.model = nn.ModuleList([cur_model for _, cur_model in enumerate(model_list)])
        self.num_seg_class = num_seg_classes
    def forward(self, segs_input, segs_images, segs_labels, train_mode, optimizer, loss_func):
        full_obj = []
        loss_one_epoch = 0
        #vis = visdom.Visdom()
        if train_mode == 'training':
            self.model.train()

            #loss_one_epoch /= len(segs_input)
            #vis.image(images[0])
            for i in range(len(segs_input)):
                output = self.model[i](segs_input[i], segs_images[i])
                loss, pnll = loss_func(segs_input[i], output)
                loss_one_epoch += loss
                #full_obj.append(output['p_prior_samples'][-1])      #no need

                #vis_input = torch.transpose(segs_input[i], 1, 2)
                #vis.scatter(vis_input[0])
                #vis_opc = torch.transpose(output['p_prior_samples'][-1], 1, 2)
                #vis.scatter(vis_opc[0])
            loss_one_epoch /= len(segs_input)

            optimizer.zero_grad()
            loss_one_epoch.backward()
            optimizer.step()

            return loss_one_epoch

        else:
            self.model.eval()

            for i, label in enumerate(segs_labels):
                with torch.no_grad():
                    output = self.model[label](segs_input[i], segs_images[i])
                    loss, pnll = loss_func(segs_input[i], output)
                loss_one_epoch += loss
                full_obj.append(output['p_prior_samples'][-1])

            loss_one_epoch /= len(segs_input)

            full_obj = torch.cat(full_obj, dim=2)
            #vis_output = torch.transpose(full_obj, 1, 2)
            #vis.scatter(vis_output[0])

            return full_obj, loss_one_epoch

