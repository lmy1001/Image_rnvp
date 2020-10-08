import torch
import torch.nn as nn
from networks.encoders import PointNetCloudEncoder, FeatureEncoder
from networks.decoders import LocalCondRNVPDecoder, GlobalRNVPDecoder
from networks.resnet import resnet18
import io
import yaml

#with no global rnvp module, and image features add as conditions in local conditional RNVP
class Conditional_RNVP(nn.Module):
    def __init__(self, **kwargs):
        super(Conditional_RNVP, self).__init__()

        self.mode = kwargs.get('usage_mode')
        self.deterministic = kwargs.get('deterministic')

        self.pc_enc_init_n_channels = kwargs.get('pc_enc_init_n_channels')
        self.pc_enc_init_n_features = kwargs.get('pc_enc_init_n_features')
        self.pc_enc_n_features = kwargs.get('pc_enc_n_features')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')
        self.g_prior_n_layers = kwargs.get('g_prior_n_layers')  # 1
        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')

        self.g_posterior_n_layers = kwargs.get('g_posterior_n_layers')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')
        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_base_var = kwargs.get('p_decoder_base_var')

        self.pc_encoder = PointNetCloudEncoder(self.pc_enc_init_n_channels,
                                               self.pc_enc_init_n_features,
                                               self.pc_enc_n_features)

        self.g0_prior_mus = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        self.g0_prior_logvars = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        with torch.no_grad():
            nn.init.normal_(self.g0_prior_mus.data, mean=0.0, std=0.033)
            nn.init.normal_(self.g0_prior_logvars.data, mean=0.0, std=0.33)

        self.g_posterior = FeatureEncoder(self.g_posterior_n_layers, self.pc_enc_n_features[-1],
                                          self.g_latent_space_size, deterministic=False,
                                          mu_weight_std=0.0033, mu_bias=0.0,
                                          logvar_weight_std=0.033, logvar_bias=0.0)

        self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                      self.p_latent_space_size, deterministic=False,
                                      mu_weight_std=0.001, mu_bias=0.0,
                                      logvar_weight_std=0.01, logvar_bias=0.0)

        self.pc_decoder = LocalCondRNVPDecoder(self.p_decoder_n_flows,
                                               self.p_decoder_n_features,
                                               self.g_latent_space_size,
                                               weight_std=0.01)

        self.g0_prior = FeatureEncoder(self.g_prior_n_layers, self.g_latent_space_size,
                                       self.g_latent_space_size, deterministic=False,
                                       mu_weight_std=0.0033, mu_bias=0.0,
                                       logvar_weight_std=0.033, logvar_bias=0.0)
        self.image_encoder = resnet18(num_classes=self.g_latent_space_size)
        self.num_gen_samples = 17

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, p_input, images, n_sampled_points=None):
        sampled_cloud_size = p_input.shape[2] if n_sampled_points is None else n_sampled_points
        self.num_gen_samples = p_input.shape[0]
        output = {}
        if self.mode == 'training':
            p_enc_features = self.pc_encoder(p_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]

            ####add image features as condition to the RNVP
            img_features = self.image_encoder(images)
            output['cond_mus'], output['cond_logvars'] = self.g0_prior(img_features)      # N * latent
            output['cond_samples'] = self.reparameterize(output['cond_mus'], output['cond_logvars'])
            ####

            output['g_prior_mus'] = [self.g0_prior_mus.expand(p_input.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(p_input.shape[0], self.g_latent_space_size)]
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]



            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'], output['g_posterior_logvars'])

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]

            buf_p = self.pc_decoder(p_input, output['cond_samples'], mode='inverse')
            output['p_prior_samples'] = buf_p[0] + [p_input]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        elif self.mode == 'predicting':
            img_features = self.image_encoder(images)
            output['cond_mus'], output['cond_logvars'] = self.g0_prior(img_features)      # N * latent
            output['cond_samples'] = self.reparameterize(output['cond_mus'], output['cond_logvars'])

            output['g_prior_mus'] = [self.g0_prior_mus.expand(self.num_gen_samples, self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(self.num_gen_samples, self.g_latent_space_size)]
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'][-1])
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                self.num_gen_samples, self.p_latent_space_size, sampled_cloud_size
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                self.num_gen_samples, self.p_latent_space_size, sampled_cloud_size
            )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]

            buf_p = self.pc_decoder(output['p_prior_samples'][0], output['cond_samples'], mode='direct')
            output['p_prior_samples'] += buf_p[0]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        return output

#
class Conditional_RNVP_with_Global_RNVP(nn.Module):
    def __init__(self, **kwargs):
        super(Conditional_RNVP_with_Global_RNVP, self).__init__()

        self.mode = kwargs.get('usage_mode')
        self.deterministic = kwargs.get('deterministic')

        self.pc_enc_init_n_channels = kwargs.get('pc_enc_init_n_channels')
        self.pc_enc_init_n_features = kwargs.get('pc_enc_init_n_features')
        self.pc_enc_n_features = kwargs.get('pc_enc_n_features')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')

        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')

        self.g_posterior_n_layers = kwargs.get('g_posterior_n_layers')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')
        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_base_var = kwargs.get('p_decoder_base_var')

        self.pc_encoder = PointNetCloudEncoder(self.pc_enc_init_n_channels,
                                               self.pc_enc_init_n_features,
                                               self.pc_enc_n_features)

        self.g0_prior_mus = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        self.g0_prior_logvars = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        with torch.no_grad():
            nn.init.normal_(self.g0_prior_mus.data, mean=0.0, std=0.033)
            nn.init.normal_(self.g0_prior_logvars.data, mean=0.0, std=0.33)

        self.g_prior = GlobalRNVPDecoder(self.g_prior_n_flows, self.g_prior_n_features,
                                         self.g_latent_space_size, weight_std=0.01)

        self.g_posterior = FeatureEncoder(self.g_posterior_n_layers, self.pc_enc_n_features[-1],
                                          self.g_latent_space_size, deterministic=False,
                                          mu_weight_std=0.0033, mu_bias=0.0,
                                          logvar_weight_std=0.033, logvar_bias=0.0)

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
            p_enc_features = self.pc_encoder(p_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]

            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'], output['g_posterior_logvars'])

            output['g_prior_mus'] = [self.g0_prior_mus.expand(p_input.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(p_input.shape[0], self.g_latent_space_size)]
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            ####add image features as condition to the RNVP
            img_features = self.image_encoder(images)
            output['cond_mus'], output['cond_logvars'] = self.g0_prior(img_features)      # N * latent
            output['cond_samples'] = self.reparameterize(output['cond_mus'], output['cond_logvars'])
            ####

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
            )]

            buf_p = self.pc_decoder(p_input, output['cond_samples'], mode='inverse')
            output['p_prior_samples'] = buf_p[0] + [p_input]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        elif self.mode == 'predicting':
            output['g_prior_mus'] = [self.g0_prior_mus.expand(images.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(images.shape[0], self.g_latent_space_size)]
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            output['g_prior_samples'] += buf_g[0]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            ####add image features as condition to the RNVP
            img_features = self.image_encoder(images)
            output['cond_mus'], output['cond_logvars'] = self.g0_prior(img_features)      # N * latent
            output['cond_samples'] = self.reparameterize(output['cond_mus'], output['cond_logvars'])
            ####

            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'][-1])
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
            )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]


            buf_p = self.pc_decoder(output['p_prior_samples'][0], output['cond_samples'], mode='direct')
            output['p_prior_samples'] += buf_p[0]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        return output

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
    def __init__(self, model, num_seg_classes):
        super(Model_of_Full_Obj, self).__init__()
        self.model = nn.ModuleList([model for _ in range(num_seg_classes)])
        self.num_seg_class = num_seg_classes
    def forward(self, segs_input, images, train_mode, optimizer, loss_func):
        full_obj = []
        loss_one_epoch = 0
        if train_mode == 'training':
            self.model.train()
            for i in range(len(segs_input)):
                output = self.model[i](segs_input[i], images)
                loss, pnll = loss_func(segs_input[i], output)

                loss_one_epoch += loss

                full_obj.append(output['p_prior_samples'][-1])

            loss_one_epoch /= len(segs_input)

            optimizer.zero_grad()
            loss_one_epoch.backward()
            optimizer.step()

        else:
            self.model.eval()
            for i in range(len(segs_input)):
                with torch.no_grad():
                    output = self.model[i](segs_input[i], images)
                    loss, pnll = loss_func(segs_input[i], output)
                loss_one_epoch += loss
                full_obj.append(output['p_prior_samples'][-1])

        full_obj = torch.cat(full_obj, dim=2)

        loss_one_epoch /= len(segs_input)
        return full_obj, loss_one_epoch


if __name__=='__main__':
    config_dir = '../image_based_model.yaml'
    with io.open(config_dir, 'r') as stream:
        config = yaml.load(stream)
    modelname='all_svr_model'
    config['model_name'] = '{0}.pkl'.format(modelname)
    config['n_epochs'] = 20

    net = Conditional_RNVP_with_Global_RNVP(**config)

    input = torch.randn(10, 3, 6)
    images = torch.randn(10, 4, 32, 32)

    output = net(input, images)
    print(output)
