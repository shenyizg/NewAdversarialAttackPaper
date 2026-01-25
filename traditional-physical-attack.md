# Traditional Deep Learning Models - Physical Attack
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning**

cs.RO

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14104v1) [paper-pdf](https://arxiv.org/pdf/2601.14104v1)

**Confidence**: 0.95

**Authors**: Tairan Huang, Qingqing Ye, Yulin Jin, Jiawei Lian, Yi Wang, Haibo Hu

**Abstract**: Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.



## **2. Guided Diffusion-based Generation of Adversarial Objects for Real-World Monocular Depth Estimation Attacks**

cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24111v1) [paper-pdf](https://arxiv.org/pdf/2512.24111v1)

**Confidence**: 0.95

**Authors**: Yongtao Chen, Yanbo Wang, Wentao Zhao, Guole Shen, Tianchen Deng, Jingchuan Wang

**Abstract**: Monocular Depth Estimation (MDE) serves as a core perception module in autonomous driving systems, but it remains highly susceptible to adversarial attacks. Errors in depth estimation may propagate through downstream decision making and influence overall traffic safety. Existing physical attacks primarily rely on texture-based patches, which impose strict placement constraints and exhibit limited realism, thereby reducing their effectiveness in complex driving environments. To overcome these limitations, this work introduces a training-free generative adversarial attack framework that generates naturalistic, scene-consistent adversarial objects via a diffusion-based conditional generation process. The framework incorporates a Salient Region Selection module that identifies regions most influential to MDE and a Jacobian Vector Product Guidance mechanism that steers adversarial gradients toward update directions supported by the pre-trained diffusion model. This formulation enables the generation of physically plausible adversarial objects capable of inducing substantial adversarial depth shifts. Extensive digital and physical experiments demonstrate that our method significantly outperforms existing attacks in effectiveness, stealthiness, and physical deployability, underscoring its strong practical implications for autonomous driving safety assessment.



## **3. Projection-based Adversarial Attack using Physics-in-the-Loop Optimization for Monocular Depth Estimation**

cs.CV

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24792v1) [paper-pdf](https://arxiv.org/pdf/2512.24792v1)

**Confidence**: 0.95

**Authors**: Takeru Kusakabe, Yudai Hirose, Mashiho Mukaida, Satoshi Ono

**Abstract**: Deep neural networks (DNNs) remain vulnerable to adversarial attacks that cause misclassification when specific perturbations are added to input images. This vulnerability also threatens the reliability of DNN-based monocular depth estimation (MDE) models, making robustness enhancement a critical need in practical applications. To validate the vulnerability of DNN-based MDE models, this study proposes a projection-based adversarial attack method that projects perturbation light onto a target object. The proposed method employs physics-in-the-loop (PITL) optimization -- evaluating candidate solutions in actual environments to account for device specifications and disturbances -- and utilizes a distributed covariance matrix adaptation evolution strategy. Experiments confirmed that the proposed method successfully created adversarial examples that lead to depth misestimations, resulting in parts of objects disappearing from the target scene.



## **4. Real-World Adversarial Attacks on RF-Based Drone Detectors**

cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2512.20712v2) [paper-pdf](https://arxiv.org/pdf/2512.20712v2)

**Confidence**: 0.95

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.



## **5. Misspecified Crame-Rao Bound for AoA Estimation at a ULA under a Spoofing Attack**

eess.SP

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16735v1) [paper-pdf](https://arxiv.org/pdf/2512.16735v1)

**Confidence**: 0.95

**Authors**: Sotiris Skaperas, Arsenia Chorti

**Abstract**: A framework is presented for analyzing the impact of active attacks to location-based physical layer authentication (PLA) using the machinery of misspecified Cramér--Rao bound (MCRB). In this work, we focus on the MCRB in the angle-of-arrival (AoA) based authentication of a single antenna user when the verifier posseses an $M$ antenna element uniform linear array (ULA), assuming deterministic pilot signals; in our system model the presence of a spoofing adversary with an arbitrary number $L$ of antenna elements is assumed. We obtain a closed-form expression for the MCRB and demonstrate that the attack introduces in it a penalty term compared to the classic CRB, which does not depend on the signal-to-noise ratio (SNR) but on the adversary's location, the array geometry and the attacker precoding vector.



## **6. Talking to the Airgap: Exploiting Radio-Less Embedded Devices as Radio Receivers**

cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15387v1) [paper-pdf](https://arxiv.org/pdf/2512.15387v1)

**Confidence**: 0.95

**Authors**: Paul Staat, Daniel Davidovich, Christof Paar

**Abstract**: Intelligent electronics are deeply embedded in critical infrastructures and must remain reliable, particularly against deliberate attacks. To minimize risks and impede remote compromise, sensitive systems can be physically isolated from external networks, forming an airgap. Yet, airgaps can still be infiltrated by capable adversaries gaining code execution. Prior research has shown that attackers can then attempt to wirelessly exfiltrate data across the airgap by exploiting unintended radio emissions. In this work, we demonstrate reversal of this link: malicious code execution on embedded devices can enable wireless infiltration of airgapped systems without any hardware modification. In contrast to previous infiltration methods that depend on dedicated sensors (e.g., microphones, LEDs, or temperature sensors) or require strict line-of-sight, we show that unmodified, sensor-less embedded devices can inadvertently act as radio receivers. This phenomenon stems from parasitic RF sensitivity in PCB traces and on-chip analog-to-digital converters (ADCs), allowing external transmissions to be received and decoded entirely in software.   Across twelve commercially available embedded devices and two custom prototypes, we observe repeatable reception in the 300-1000 MHz range, with detectable signal power as low as 1 mW. To this end, we propose a systematic methodology to identify device configurations that foster such radio sensitivities and comprehensively evaluate their feasibility for wireless data reception. Exploiting these sensitivities, we demonstrate successful data reception over tens of meters, even in non-line-of-sight conditions and show that the reception sensitivities accommodate data rates of up to 100 kbps. Our findings reveal a previously unexplored command-and-control vector for air-gapped systems while challenging assumptions about their inherent isolation. [shortened]



## **7. PHANTOM: PHysical ANamorphic Threats Obstructing Connected Vehicle Mobility**

cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.19711v1) [paper-pdf](https://arxiv.org/pdf/2512.19711v1)

**Confidence**: 0.95

**Authors**: Md Nahid Hasan Shuvo, Moinul Hossain

**Abstract**: Connected autonomous vehicles (CAVs) rely on vision-based deep neural networks (DNNs) and low-latency (Vehicle-to-Everything) V2X communication to navigate safely and efficiently. Despite their advances, these systems remain vulnerable to physical adversarial attacks. In this paper, we introduce PHANTOM (PHysical ANamorphic Threats Obstructing connected vehicle Mobility), a novel framework for crafting and deploying perspective-dependent adversarial examples using \textit{anamorphic art}. PHANTOM exploits geometric distortions that appear natural to humans but are misclassified with high confidence by state-of-the-art object detectors. Unlike conventional attacks, PHANTOM operates in black-box settings without model access and demonstrates strong transferability across four diverse detector architectures (YOLOv5, SSD, Faster R-CNN, and RetinaNet). Comprehensive evaluation in CARLA across varying speeds, weather conditions, and lighting scenarios shows that PHANTOM achieves over 90\% attack success rate under optimal conditions and maintains 60-80\% effectiveness even in degraded environments. The attack activates within 6-10 meters of the target, providing insufficient time for safe maneuvering. Beyond individual vehicle deception, PHANTOM triggers network-wide disruption in CAV systems: SUMO-OMNeT++ co-simulation demonstrates that false emergency messages propagate through V2X links, increasing Peak Age of Information by 68-89\% and degrading safety-critical communication. These findings expose critical vulnerabilities in both perception and communication layers of CAV ecosystems.



## **8. Exposing Vulnerabilities in Counterfeit Prevention Systems Utilizing Physically Unclonable Surface Features**

cs.CR

15 pages; This work builds on arXiv:2408.02221 [cs.CR]

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.09150v1) [paper-pdf](https://arxiv.org/pdf/2512.09150v1)

**Confidence**: 0.95

**Authors**: Anirudh Nakra, Nayeeb Rashid, Chau-Wai Wong, Min Wu

**Abstract**: Counterfeit products pose significant risks to public health and safety through infiltrating untrusted supply chains. Among numerous anti-counterfeiting techniques, leveraging inherent, unclonable microscopic irregularities of paper surfaces is an accurate and cost-effective solution. Prior work of this approach has focused on enabling ubiquitous acquisition of these physically unclonable features (PUFs). However, we will show that existing authentication methods relying on paper surface PUFs may be vulnerable to adversaries, resulting in a gap between technological feasibility and secure real-world deployment. This gap is investigated through formalizing an operational framework for paper-PUF-based authentication. Informed by this framework, we reveal system-level vulnerabilities across both physical and digital domains, designing physical denial-of-service and digital forgery attacks to disrupt proper authentication. The effectiveness of the designed attacks underscores the strong need for security countermeasures for reliable and resilient authentication based on paper PUFs. The proposed framework further facilitates a comprehensive, stage-by-stage security analysis, guiding the design of future counterfeit prevention systems. This analysis delves into potential attack strategies, offering a foundational understanding of how various system components, such as physical features and verification processes, might be exploited by adversaries.



## **9. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

cs.CV

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.16020v2) [paper-pdf](https://arxiv.org/pdf/2511.16020v2)

**Confidence**: 0.95

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.



## **10. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

cs.CV

14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2411.10029v2) [paper-pdf](https://arxiv.org/pdf/2411.10029v2)

**Confidence**: 0.95

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.



## **11. Visual Adversarial Attacks and Defenses in the Physical World: A Survey**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2211.01671v6) [paper-pdf](https://arxiv.org/pdf/2211.01671v6)

**Confidence**: 0.95

**Authors**: Xingxing Wei, Bangzheng Pu, Shiji Zhao, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they remain vulnerable to adversarial examples. Adversarial attacks in computer vision can be categorized into digital attacks and physical attacks based on their different forms. Compared to digital attacks, which generate perturbations in digital pixels, physical attacks are more practical in real-world settings. Due to the serious security risks posed by physically adversarial examples, many studies have been conducted to evaluate the physically adversarial robustness of DNNs in recent years. In this paper, we provide a comprehensive survey of current physically adversarial attacks and defenses in computer vision. We establish a taxonomy by organizing physical attacks according to attack tasks, attack forms, and attack methods. This approach offers readers a systematic understanding of the topic from multiple perspectives. For physical defenses, we categorize them into pre-processing, in-processing, and post-processing for DNN models to ensure comprehensive coverage of adversarial defenses. Based on this survey, we discuss the challenges facing this research field and provide an outlook on future directions.



## **12. Physical ID-Transfer Attacks against Multi-Object Tracking via Adversarial Trajectory**

cs.CV

Accepted to Annual Computer Security Applications Conference (ACSAC) 2024

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01934v1) [paper-pdf](https://arxiv.org/pdf/2512.01934v1)

**Confidence**: 0.95

**Authors**: Chenyi Wang, Yanmao Man, Raymond Muller, Ming Li, Z. Berkay Celik, Ryan Gerdes, Jonathan Petit

**Abstract**: Multi-Object Tracking (MOT) is a critical task in computer vision, with applications ranging from surveillance systems to autonomous driving. However, threats to MOT algorithms have yet been widely studied. In particular, incorrect association between the tracked objects and their assigned IDs can lead to severe consequences, such as wrong trajectory predictions. Previous attacks against MOT either focused on hijacking the trackers of individual objects, or manipulating the tracker IDs in MOT by attacking the integrated object detection (OD) module in the digital domain, which are model-specific, non-robust, and only able to affect specific samples in offline datasets. In this paper, we present AdvTraj, the first online and physical ID-manipulation attack against tracking-by-detection MOT, in which an attacker uses adversarial trajectories to transfer its ID to a targeted object to confuse the tracking system, without attacking OD. Our simulation results in CARLA show that AdvTraj can fool ID assignments with 100% success rate in various scenarios for white-box attacks against SORT, which also have high attack transferability (up to 93% attack success rate) against state-of-the-art (SOTA) MOT algorithms due to their common design principles. We characterize the patterns of trajectories generated by AdvTraj and propose two universal adversarial maneuvers that can be performed by a human walker/driver in daily scenarios. Our work reveals under-explored weaknesses in the object association phase of SOTA MOT systems, and provides insights into enhancing the robustness of such systems.



## **13. The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.00765v2) [paper-pdf](https://arxiv.org/pdf/2512.00765v2)

**Confidence**: 0.95

**Authors**: Haojie Ji, Te Hu, Haowen Li, Long Jin, Chongshi Xin, Yuchi Yao, Jiarui Xiao

**Abstract**: Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.



## **14. Adversarial Patch Attacks on Vision-Based Cargo Occupancy Estimation via Differentiable 3D Simulation**

cs.CV

9 pages, 5 figures, 1 algorithm

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19254v1) [paper-pdf](https://arxiv.org/pdf/2511.19254v1)

**Confidence**: 0.95

**Authors**: Mohamed Rissal Hedna, Sesugh Samuel Nder

**Abstract**: Computer vision systems are increasingly adopted in modern logistics operations, including the estimation of trailer occupancy for planning, routing, and billing. Although effective, such systems may be vulnerable to physical adversarial attacks, particularly adversarial patches that can be printed and placed on interior surfaces. In this work, we study the feasibility of such attacks on a convolutional cargo-occupancy classifier using fully simulated 3D environments. Using Mitsuba 3 for differentiable rendering, we optimize patch textures across variations in geometry, lighting, and viewpoint, and compare their effectiveness to a 2D compositing baseline. Our experiments demonstrate that 3D-optimized patches achieve high attack success rates, especially in a denial-of-service scenario (empty to full), where success reaches 84.94 percent. Concealment attacks (full to empty) prove more challenging but still reach 30.32 percent. We analyze the factors influencing attack success, discuss implications for the security of automated logistics pipelines, and highlight directions for strengthening physical robustness. To our knowledge, this is the first study to investigate adversarial patch attacks for cargo-occupancy estimation in physically realistic, fully simulated 3D scenes.



## **15. Robust Physical Adversarial Patches Using Dynamically Optimized Clusters**

cs.CV

Supplementary material available at: https://drive.google.com/drive/folders/1Yntcc9CARdbvoJJ51cyUm1DWGSvU9X4V?usp=drive_link

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18656v1) [paper-pdf](https://arxiv.org/pdf/2511.18656v1)

**Confidence**: 0.95

**Authors**: Harrison Bagley, Will Meakin, Simon Lucey, Yee Wei Law, Tat-Jun Chin

**Abstract**: Physical adversarial attacks on deep learning systems is concerning due to the ease of deploying such attacks, usually by placing an adversarial patch in a scene to manipulate the outcomes of a deep learning model. Training such patches typically requires regularization that improves physical realizability (e.g., printability, smoothness) and/or robustness to real-world variability (e.g. deformations, viewing angle, noise). One type of variability that has received little attention is scale variability. When a patch is rescaled, either digitally through downsampling/upsampling or physically through changing imaging distances, interpolation-induced color mixing occurs. This smooths out pixel values, resulting in a loss of high-frequency patterns and degrading the adversarial signal. To address this, we present a novel superpixel-based regularization method that guides patch optimization to scale-resilient structures. Our ap proach employs the Simple Linear Iterative Clustering (SLIC) algorithm to dynamically cluster pixels in an adversarial patch during optimization. The Implicit Function Theorem is used to backpropagate gradients through SLIC to update the superpixel boundaries and color. This produces patches that maintain their structure over scale and are less susceptible to interpolation losses. Our method achieves greater performance in the digital domain, and when realized physically, these performance gains are preserved, leading to improved physical performance. Real-world performance was objectively assessed using a novel physical evaluation protocol that utilizes screens and cardboard cut-outs to systematically vary real-world conditions.



## **16. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

cs.CV

AAAI 2026

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.14386v3) [paper-pdf](https://arxiv.org/pdf/2511.14386v3)

**Confidence**: 0.95

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.



## **17. Post-Quantum Cryptography for Intelligent Transportation Systems: An Implementation-Focused Review**

cs.CR

This is a preprint version of a manuscript currently under peer review. This version has not undergone peer review and may differ from the final published version

**SubmitDate**: 2026-01-03    [abs](http://arxiv.org/abs/2601.01068v1) [paper-pdf](https://arxiv.org/pdf/2601.01068v1)

**Confidence**: 0.85

**Authors**: Abdullah Al Mamun, Akid Abrar, Mizanur Rahman, M Sabbir Salek, Mashrur Chowdhury

**Abstract**: As quantum computing advances, the cryptographic algorithms that underpin confidentiality, integrity, and authentication in Intelligent Transportation Systems (ITS) face increasing vulnerability to quantum-enabled attacks. To address these risks, governments and industry stakeholders are turning toward post-quantum cryptography (PQC), a class of algorithms designed to resist adversaries equipped with quantum computing capabilities. However, existing studies provide limited insight into the implementation-focused aspects of PQC in the ITS domain. This review fills that gap by evaluating the readiness of vehicular communication and security standards for PQC adoption. It examines in-vehicle networks and vehicle-to-everything (V2X) interfaces, while also investigating vulnerabilities at the physical layer, primarily exposure to side-channel and fault injection attacks. The review identifies thirteen research gaps reflecting non-PQC-ready standards, constraints in embedded implementation and hybrid cryptography, interoperability and certificate-management barriers, lack of real-world PQC deployment data in ITS, and physical-attack vulnerabilities in PQC-enabled vehicular communication. Future research directions include updating vehicular communication and security standards, optimizing PQC for low-power devices, enhancing interoperability and certificate-management frameworks for PQC integration, conducting real-world evaluations of PQC-enabled communication and control functions across ITS deployments, and strengthening defenses against AI-assisted physical attacks. A phased roadmap is presented, aligning PQC deployment with regulatory, performance, and safety requirements, thereby guiding the secure evolution of ITS in the quantum computing era.



