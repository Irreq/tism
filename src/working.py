M = Modulation(frequency=2e3, bitrate=5, amplitude=0.5)

send_data = [1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1]

x = M.modulate(send_data)

def add_white_noise(sig, k):

    if not 0 <= k <= 1:
        print(f"K must be within 1 and 0 not {k}, k will now equal to 0.1")
        k = 0.1

    n = len(sig)

    white = np.array([np.random.random()*2-1 for i in range(n)]) * k

    mixed = white + sig * (1-k)

    if max(abs(mixed)) > 1:

      mixed /= max(abs(mixed))

    return mixed




Fs = int(M.samplingrate/2)

z = np.zeros(Fs).tolist()

z.extend(x.tolist())
z.extend(np.zeros(Fs).tolist())

x = z

x = add_white_noise(np.array(x), 0.01)

import matplotlib.pyplot as plt

def plut(k):
  l = 0

  for i in k:
    x, segments = i

    plt.subplot(211+l)
    plt.title('Signal')
    plt.plot(x)

    for s_lim in segments:
                plt.axvline(x=s_lim[0], color='red')
                plt.axvline(x=s_lim[1], color='red')

    # plt.subplot(212)
    # plt.title('Silence Removed')
    # plt.plot(x[segments[0][0]:segments[0][1]])
    # plt.show()

    l+=1

plt.plot(x)



len(x)


segments = silence_removal(x, Fs, 0.02, 0.09, smooth_window = 0.6, weight = 0.4, plot = False)

segments = [[int(i[0]*Fs), int(i[1]*Fs)] for i in segments]

d = [0 if -0.8 < i < 0.8 else i for i in x[segments[0][0]:segments[0][1]]]
g = np.zeros(Fs).tolist()

g.extend(d)
g.extend(np.zeros(Fs).tolist())
# print(d)

signal = np.array(g)

signal = add_white_noise(signal, 0.01)

# plt.plot(signal)
# plt.show()

segments2 = silence_removal(signal, Fs, 0.01, 0.1, smooth_window = 0.20, weight = 0.5, plot = False)

print(segments2)

segments2 = [[int(i[0]*Fs), int(i[1]*Fs)] for i in segments2]

plut([[x, segments],[signal, segments2]])

avg = np.mean([i[1]-i[0] for i in segments2])


bits = []
size = len(segments2)
for i in range(size):

  if i == size-1:
    break

  avg += segments2[i+1][0] - segments2[i][1]

  avg /= 2

for i in range(size):

  lol = segments2[i][1]-segments2[i][0]

  if lol > avg:
    bits.extend([1,1])
  else:
    bits.extend([1])

  if i < size-1:
    if segments2[i+1][0] - segments2[i][1] > avg:
      bits.extend([0,0])
    else:
      bits.extend([0])








# avg = np.mean(bits)

print('test {}'.format(str(bits)))
print('True {}'.format(str(send_data)))
