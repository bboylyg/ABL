import torch


def min_max_normalization(x):
	x_min = torch.min(x)
	x_max = torch.max(x)
	norm = (x - x_min) / (x_max - x_min)
	return norm


class ABLAnalysis():
	def __init__(self):
		# Based on https://github.com/bboylyg/ABL/blob/main/backdoor_isolation.py
		return

	def compute_loss_value(self, data, model_ascent):
		# Calculate loss value per example
		# Define loss function
		if opt.cuda:
			criterion = nn.CrossEntropyLoss().cuda()
		else:
			criterion = nn.CrossEntropyLoss()

		model_ascent.eval()
		losses_record = []

		example_data_loader = DataLoader(dataset=poisoned_data,
											batch_size=1,
											shuffle=False,
											)

		for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
			if opt.cuda:
				img = img.cuda()
				target = target.cuda()

			with torch.no_grad():
				output = model_ascent(img)
				loss = criterion(output, target)
				# print(loss.item())

			losses_record.append(loss.item())

		losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in ascending order

		# Show the lowest 10 loss values
		losses_record_arr = np.array(losses_record)
		print('Top ten loss value:', losses_record_arr[losses_idx[:10]])

		return losses_idx

	def isolate_data(self, data, losses_idx):
		# Initialize lists
		other_examples = []
		isolation_examples = []

		cnt = 0
		ratio = opt.isolation_ratio

		example_data_loader = DataLoader(dataset=poisoned_data,
										 batch_size=1,
										 shuffle=False,
										 )
		# print('full_poisoned_data_idx:', len(losses_idx))
		perm = losses_idx[0: int(len(losses_idx) * ratio)]

		for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
			img = img.squeeze()
			target = target.squeeze()
			img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
			target = target.cpu().numpy()

			# Filter the examples corresponding to losses_idx
			if idx in perm:
				isolation_examples.append((img, target))
				cnt += 1
			else:
				other_examples.append((img, target))

		# Save data
		if opt.save:
			data_path_isolation = os.path.join(opt.isolate_data_root,
											   "{}_isolation{}%_examples.npy".format(opt.model_name,
																					 opt.isolation_ratio * 100))
			data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}%_examples.npy".format(opt.model_name,
																									100 - opt.isolation_ratio * 100))
			if os.path.exists(data_path_isolation):
				raise ValueError('isolation data already exists')
			else:
				# save the isolation examples
				np.save(data_path_isolation, isolation_examples)
				np.save(data_path_other, other_examples)

		print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
		print('Finish collecting {} other examples: '.format(len(other_examples)))




