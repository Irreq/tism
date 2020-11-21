from normal_dist import KernelGenerator, constrained_sum_sample_nonneg

import numpy as np





class Patient(object):

    def __init__(self):

        self.id = 0 # Generic

        self.patients = []

        self.patient_structure = {



            # Social

            "age" : [0.1754, 0.1106, 0.3937, 0.1167], # 1 is equivalent of 113 years CIA World Factbook (2018 est.)


            # location

            "travel_difficulty" : [0.66, 0.206, 0.128, 0.005], # 1 is equivalent to > 30 minutes based on data from https://www.regionfakta.com/Skane-lan/Samhallets-service/Avstand-till-vardcentral/ -- 2014

            "socio_economic_vulnerability" : [0.041, 0.169, 0.327, 0.235, 0.228], # General population (SCB, 2017)

            # medical treatment

            "medical_conditions" : [0.9, 0.09, 0.01], # fictional

            "severity" : [0.9, 0.09, 0.01],

            "treatment_time" : [0.9, 0.09, 0.01],

            "treatment_frequency" : [0.9, 0.09, 0.01],

            "burden_quality" : [0.9, 0.09, 0.01], # how bad the patient is with the doctor fictional



        }

        self.invoked = False

    def invoke(self):

        if self.invoked:
            return

        kg = KernelGenerator()

        kg.start()

        for i in self.patient_structure.keys():
            self.patient_structure[i] = kg.fastgen(self.patient_structure[i])

        self.invoked = True

    def generate(self, n):

        if not self.invoked:
            self.invoke()

        patient_batch = []

        order = [
                 "age",
                 "burden_quality",
                 "travel_difficulty",
                 "socio_economic_vulnerability",
                 "medical_conditions",
                 "severity",
                 "treatment_time",
                 "treatment_frequency"
                 ]


        size = len(order)

        for i in range(n):

            # if i % 20 == 0:
            #     print(i/n*100)

            patient = []

            for i, tag in enumerate(order):

                value = np.random.choice(self.patient_structure[tag])

                if i > 0:
                    value = (value + 3*patient[-1]) / 4

                patient.append(round(value,3))

            patient_batch.append(np.array(patient))



        self.patients.append(patient_batch)

        return patient_batch


    def retrieve(self):

        return self.patients






def dmp():


        """ Example showing a biased distribution """

        import matplotlib.pyplot as plt

        # I have chosen two distributions
        # her, but you could use whatever
        # you wan't

        generic_bias_instructions = {

            "medical_conditions" : [[0.0, 0.1],
                                    [0.72, 0.06],
                                    [1.0, 0.4],
                                    [0.35, 0.1],],

            "random_distribution" : [[0.0, 0.1],
                                     [0.72, 0.1],
                                     [1.0, 0.4],
                                     [0.35, 0.1],],

        }

        generic_distribution = {

            "travel_distance" : [0.66, 0.206, 0.128, 0.005],


        }



        resolution = 500

        kg = KernelGenerator(size=resolution)

        kg.start()

        kg.fastgen()


        for i in generic_distribution.keys():
            generic_distribution[i] = kg.fastgen(generic_distribution[i])

        # # bias = kg.addbias(generic_bias_instructions)
        #
        # bias = kg.getbias()
        #
        # nd = bias["normal_distribution"]
        #
        #
        # regular_distribution = [0.66, 0.206, 0.128, 0.005]
        #
        # window = 1 / len(regular_distribution)
        #
        # nomdist = kg.setwindow(0,window,nd)
        #
        #
        #
        #
        #
        # # print(len(nomdist))
        #
        # print('elements : {}\nmean : {}\nmax : {}\nmin : {}'.format(len(nomdist),np.mean(np.array(nomdist)), max(nomdist), min(nomdist)))
        #
        #
        #
        #
        # final_distribution = []
        #
        # for i, value in enumerate(regular_distribution):
        #
        #     data = [i*window+np.random.choice(nomdist) for _ in range(int(value*resolution))]
        #
        #     final_distribution.extend(data)
        #
        #     # lower = i*window
        #     #
        #     # upper = (i+1)*window
        #     #
        #     # print(f"{i*window} - {(i+1)*window}")
        #
        # c = np.array(final_distribution)
        #
        # print(max(c))
        #
        # plt.subplot(211)
        # plt.plot(c)
        #
        # plt.subplot(212)
        # plt.hist(c, bins=np.linspace(0,1,10))
        #
        # plt.show()





        doctors = 5


def doctorgeneration(n, combinedpatients):

    doctor_distribution = constrained_sum_sample_nonneg(n, combinedpatients)



    mn = int(sum(doctor_distribution) / len(doctor_distribution))

    for i in range(n):

        doctor_distribution.sort()

        if doctor_distribution[-1] > 1.2*mn:

            doctor_distribution[-1] -= mn

            doctor_distribution[0] += mn


    doctors = {i:{'n':doctor_distribution[i]} for i in range(len(doctor_distribution))}

    p = Patient()

    p.invoke()

    final = []

    for id in doctors:
        # doctors[id] = {'patients':np.array(p.generate(doctors[id]['n'])),**doctors[id]}
        # doctors[id] = np.array(p.generate(doctors[id]['n']))

        final.extend(np.array(p.generate(doctors[id]['n'])))

    doctors = np.array(final)

    # doctors = np.array([np.array(p.generate(doctors[id]['n'])) for id in doctors])

    return doctors


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    """
    Task: Finding which doctor that has the most and least work
    """



    doctors = 700

    patients = 70
    patients = doctors

    doctors = doctorgeneration(doctors, patients)

    print(doctors)
    print(type(doctors))
    print(doctors.shape)

    x_train = doctors[:patients]

    print(x_train)

    a, b = 7, 2

    print(len(np.linspace(0,len(x_train[:, a]))))

    print(len(x_train[:, a]))




    # plt.figure(figsize=(5, 4))

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    ax.set_title('Full view')
    ax.scatter(x_train[:, a], x_train[:, b])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])


    ax2.set_title('Truncated view')
    ax2.scatter(x_train[:, a], np.linspace(0,1,num=len(x_train[:, a])))
    ax2.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    # fig, ax = plt.subplots()
    # plt.subplot(211)
    # plt.scatter(x_train[:, a], x_train[:, b])
    # plt.subplot(212)
    # plt.scatter(x_train[:, a], np.linspace(0,1,num=len(x_train[:, a])))
    # plt.colorbar(ticks=[0, 1, 2], format=formatter)
    # plt.title('Iris Dataset Scatterplot')
    # plt.xlabel(iris.feature_names[x_index])
    # plt.ylabel(iris.feature_names[y_index])



    plt.tight_layout()
    plt.show()
