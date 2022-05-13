import torch


class ILFOAttack:
    def __init__(self, model, pert_lambda, attack_norm, device, max_iter):
        super(ILFOAttack, self).__init__()
        self.device = device
        self.model = model.train()
        self.norm = attack_norm
        self.pert_lambda = pert_lambda
        self.max_iter = max_iter

        self.gate_criterion = torch.nn.MSELoss()
        self.box_min = 0.0
        self.box_max = 1.0

        self.relu = torch.nn.ReLU()

    def transform(self, images, tf_log, index, task_name):
        perturbation = torch.zeros_like(images)
        perturbation.requires_grad = True
        optimizer = torch.optim.SGD([perturbation], lr=0.01)
        for i in range(self.max_iter):
            adv_images = perturbation + images
            adv_images = adv_images.to(device=self.device)

            perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), self.norm, dim=1)
            loss_perturb = torch.mean(perturb)

            _, _, gate_probs = self.model(adv_images, self.device)

            gates = (gate_probs > 0.5).float().sum()

            gate_probs = self.relu(0.5 - gate_probs)
            loss_gate = gate_probs.sum(1).mean()

            loss = loss_gate * self.pert_lambda + loss_perturb

            tf_log.log_value(task_name + '_gates:' + str(index) + '_' + str(i), gates.item())
            tf_log.log_value(task_name + '_gate loss:' + str(index) + '_' + str(i), loss_gate.item())
            tf_log.log_value(task_name + '_per loss:' + str(index) + '_' + str(i), loss_perturb.item())

            loss.backward()
            optimizer.step()
        return perturbation
        # if self.norm == float('inf'):
        #     images = torch.clamp(perturbation, -self.per_size, self.per_size) + images
        # else:
        #     ori_shape = perturbation.shape
        #     per_norm = perturbation.reshape([len(images), -1]).norm(p=self.norm, dim=-1)
        #     per_norm = per_norm.reshape([len(images), -1])
        #     perturbation = perturbation.reshape([len(images), -1]) / per_norm * self.per_size
        #     perturbation = perturbation.reshape(ori_shape)
        #     images = perturbation + images
        # return torch.clamp(images, 0, 1).detach()

