import random
import torch
import torch.nn.functional as F



#The below function will optimize the gen_movie_embeds
#expects input of form [user_amt, emb_dim] x [user_amt, gen_vids_per_user, emb_dim]
# to test a different model, modify the below function and ensure the parameter tables align

def mini_model(model, user_embeddings, gen_movie_embeds):
    #first need to reshape user_embeddings
    u_emb = user_embeddings.unsqueeze(axis = 1)
    amt_repeat = gen_movie_embeds.shape[1]
    u_emb = u_emb.repeat(1, amt_repeat, 1)
    u_emb = F.normalize(u_emb, dim = -1)
    gen_movie_embeds = F.normalize(gen_movie_embeds, dim = -1)
    x = torch.concat([u_emb.detach(), gen_movie_embeds], axis=-1)
    x = x.view(-1, 2 * model.emb_dim)

    for i, fc in enumerate(model.fc_stack):
        x = fc(x)
        if i != len(model.fc_stack) - 1:
            x = F.relu(x)
    return F.sigmoid(x)

def GVRecall(model, data_loader, user_set = None, amt = 5, steps = 50, user_amt = 5):
    if (not user_set):
        random.seed(42)
        user_set = random.choices(list(range(model.user_embedding.weight.shape[0])), k=user_amt)
    model.eval()
    user_set_t = torch.Tensor(user_set).long()
    user_amt = len(user_set)
    user_embeddings = model.user_embedding(user_set_t)

    #no bias, no regularization

    gen_movie_embeds = torch.randn((user_amt, amt, model.emb_dim)).requires_grad_(True) #assume no bias
    control_sample = torch.randn((user_amt, amt, model.emb_dim))

    gen_optim = torch.optim.Adam([gen_movie_embeds], lr = 0.1)
    for step in range(steps):
        gen_optim.zero_grad(())
        pred = mini_model(model, user_embeddings, gen_movie_embeds)
        loss = -1 * torch.sum(pred)
        loss.backward()
        gen_optim.step()
    ideal_perm = mini_model(model, user_embeddings, gen_movie_embeds)


    all_movie_embeddings = model.movie_embedding.weight #N, emb_dim
    all_emb_n = F.normalize(all_movie_embeddings, dim = -1)
    gen_emb_n = F.normalize(gen_movie_embeds.view(-1, model.emb_dim), dim = -1)
    # print(all_emb_n.shape, gen_emb_n.shape)
    match = torch.argmax(gen_emb_n @ all_emb_n.T, dim = -1)
    closest_real_emb = torch.stack([all_emb_n[i.item()] for i in match])
    # print(match.shape, closest_real_emb.shape)
    real_perm = mini_model(model, user_embeddings, closest_real_emb.view(user_amt, amt, -1))

    control_perm = mini_model(model, user_embeddings, control_sample)


    all_embeds = all_movie_embeddings.unsqueeze(0).repeat(user_amt, 1, 1)
    best_perm = mini_model(model, user_embeddings, all_embeds)
    best_perm = torch.topk(best_perm.view(user_amt, -1), axis = -1, k = amt).values

    print(*[torch.mean(x).item() for x in [ideal_perm, real_perm, control_perm, best_perm]])

    print(ideal_perm.view(-1, amt))
    print(real_perm.view(-1, amt))
    print(control_perm.view(-1, amt))
    print(best_perm)

    # print(torch.sort(best_perm, dim = 0))

