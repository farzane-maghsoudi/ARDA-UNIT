class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(Discriminator, self).__init__() 

        # proposed Encoder
        up1 = [nn.Conv2d(256, 128, 1, bias=True), nn.ReflectionPad2d(4), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        up2 = [nn.Conv2d(512, 128, 1, bias=True), nn.ReflectionPad2d(2), nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)]
        up3 = [nn.Conv2d(1024, 128, 1, bias=True), nn.ReflectionPad2d(1), nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)]
        
        enc1 = [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True)), nn.LeakyReLU(0.2, True)]
        enc2 = [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True)), nn.LeakyReLU(0.2, True)]
        enc3 = [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True)), nn.LeakyReLU(0.2, True)]

        #Proposed adaptive feature fution.
        self.softmaxAFF = nn.Softmax(3)
        AFF1 = [nn.ReflectionPad2d(1),
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0, bias=True),
                nn.InstanceNorm2d(128)]
        AFF2 = [nn.ReflectionPad2d(1),
                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0, bias=True)]
        AFF = [nn.Conv2d(3*128, 128, kernel_size=1, stride=1, padding=0, bias=True),
               nn.ReflectionPad2d(1),
               nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True)]
        
        
        # Class Activation Map
        mult = 2 ** (1)
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * mult * 2, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))


        #Discriminator
        Dis0_0 = []
        for i in range(2, n_layers - 4):   # 1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)
            Dis0_0 += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)
        Dis0_1 = [nn.ReflectionPad2d(1),     #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)
        self.conv0 = nn.utils.spectral_norm(   #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        
        Dis1_0 = []
        for i in range(n_layers - 4, n_layers - 2):   # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 94
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 94 + 96 = 190
                nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(   #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.pad = nn.ReflectionPad2d(1)

        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)
        
        self.enc1 = nn.Sequential(*enc1)
        self.enc2 = nn.Sequential(*enc2)
        self.enc3 = nn.Sequential(*enc3)
        self.AFF1 = nn.Sequential(*AFF1)
        self.AFF2 = nn.Sequential(*AFF2)
        self.AFF = nn.Sequential(*AFF)
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)
        self.up3 = nn.Sequential(*up3)

    def forward(self, input):
        aff1, aff2, aff3 = feature_pretrain(input)

        aff1 = self.up1(aff1)
        aff2 = self.up2(aff2)
        aff3 = self.up3(aff3)

        aff1 = self.enc1(aff1)
        aff2 = self.enc1(aff2)
        aff3 = self.enc1(aff3)
        
        aff1 = aff1 * self.softmaxAFF(self.AFF1(aff1))
        aff2 = aff2 * self.softmaxAFF(self.AFF2(aff2))

        x_0 = x = self.AFF(torch.cat([aff1, aff2, aff3], 1))

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        x = torch.cat([x, x], 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc.parameters())[0]
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = self.conv1x1(x)

        x = self.lamda*x + x_0
        x = self.leaky_relu(x)
        
        heatmap = torch.sum(x, dim=1, keepdim=True)
        z = x

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)
        
        return out0, out1, cam_logit, heatmap, z

class pre_model(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.selected_out = OrderedDict()
        self.pretrained = models.resnet152(pretrained=True).cuda()
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return self.selected_out

def feature_pretrain(x):
    x = resize2d(x, (224,224))
    model = pre_model(output_layers = [0,1,2,3,4,5,6,7,8,9])
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    layerout = model(x)
    layer1out = layerout['layer1']
    layer2out = layerout['layer2']
    layer3out = layerout['layer3']

    return layer1out, layer2out, layer3out
