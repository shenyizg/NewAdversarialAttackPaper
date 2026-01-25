# 传统深度学习模型 - 物理攻击
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning**

现实世界强化学习中的扩散引导后门攻击 cs.RO

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14104v1) [paper-pdf](https://arxiv.org/pdf/2601.14104v1)

**Confidence**: 0.95

**Authors**: Tairan Huang, Qingqing Ye, Yulin Jin, Jiawei Lian, Yi Wang, Haibo Hu

**Abstract**: Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.

摘要: 后门攻击在强化学习（RL）策略中嵌入隐藏的恶意行为，并在测试时使用触发器激活它们。大多数现有攻击仅在仿真环境中验证，而它们在现实世界机器人系统中的有效性尚不明确。在物理部署中，安全约束控制流程（如速度限制、动作平滑和碰撞避免）会抑制异常动作，导致传统后门攻击大幅衰减。我们研究了这一先前被忽视的问题，并提出了一个面向现实世界RL的扩散引导后门攻击框架（DGBA）。我们设计了放置在地面的小型可打印视觉补丁触发器，并使用条件扩散模型生成它们，该模型能在现实世界视觉变化下产生多样化的补丁外观。我们将机器人控制栈视为黑盒系统。进一步引入了基于优势的投毒策略，仅在决策关键的训练状态中注入触发器。我们在TurtleBot3移动机器人上评估了该方法，证明了在保持正常任务性能的同时，能够可靠激活目标攻击。演示视频和代码可在补充材料中获取。



## **2. Guided Diffusion-based Generation of Adversarial Objects for Real-World Monocular Depth Estimation Attacks**

基于引导扩散的真实世界单目深度估计攻击对抗物体生成方法 cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.24111v1) [paper-pdf](https://arxiv.org/pdf/2512.24111v1)

**Confidence**: 0.95

**Authors**: Yongtao Chen, Yanbo Wang, Wentao Zhao, Guole Shen, Tianchen Deng, Jingchuan Wang

**Abstract**: Monocular Depth Estimation (MDE) serves as a core perception module in autonomous driving systems, but it remains highly susceptible to adversarial attacks. Errors in depth estimation may propagate through downstream decision making and influence overall traffic safety. Existing physical attacks primarily rely on texture-based patches, which impose strict placement constraints and exhibit limited realism, thereby reducing their effectiveness in complex driving environments. To overcome these limitations, this work introduces a training-free generative adversarial attack framework that generates naturalistic, scene-consistent adversarial objects via a diffusion-based conditional generation process. The framework incorporates a Salient Region Selection module that identifies regions most influential to MDE and a Jacobian Vector Product Guidance mechanism that steers adversarial gradients toward update directions supported by the pre-trained diffusion model. This formulation enables the generation of physically plausible adversarial objects capable of inducing substantial adversarial depth shifts. Extensive digital and physical experiments demonstrate that our method significantly outperforms existing attacks in effectiveness, stealthiness, and physical deployability, underscoring its strong practical implications for autonomous driving safety assessment.

摘要: 单目深度估计（MDE）是自动驾驶系统中的核心感知模块，但其极易受到对抗攻击。深度估计误差可能通过下游决策传播并影响整体交通安全。现有物理攻击主要依赖基于纹理的补丁，这些方法存在严格的放置限制且真实感有限，从而降低了其在复杂驾驶环境中的有效性。为克服这些限制，本研究提出了一种免训练的生成式对抗攻击框架，通过基于扩散的条件生成过程生成自然且与场景一致的对抗物体。该框架包含一个显著区域选择模块，用于识别对MDE影响最大的区域，以及一个雅可比向量积引导机制，将对抗梯度引导至预训练扩散模型支持的更新方向。这种设计能够生成物理上合理的对抗物体，能够引发显著的对抗性深度偏移。大量数字和物理实验表明，我们的方法在有效性、隐蔽性和物理可部署性方面显著优于现有攻击方法，凸显了其对自动驾驶安全评估的重要实践意义。



## **3. Projection-based Adversarial Attack using Physics-in-the-Loop Optimization for Monocular Depth Estimation**

基于投影的对抗攻击：采用物理闭环优化的单目深度估计方法 cs.CV

**SubmitDate**: 2025-12-31    [abs](http://arxiv.org/abs/2512.24792v1) [paper-pdf](https://arxiv.org/pdf/2512.24792v1)

**Confidence**: 0.95

**Authors**: Takeru Kusakabe, Yudai Hirose, Mashiho Mukaida, Satoshi Ono

**Abstract**: Deep neural networks (DNNs) remain vulnerable to adversarial attacks that cause misclassification when specific perturbations are added to input images. This vulnerability also threatens the reliability of DNN-based monocular depth estimation (MDE) models, making robustness enhancement a critical need in practical applications. To validate the vulnerability of DNN-based MDE models, this study proposes a projection-based adversarial attack method that projects perturbation light onto a target object. The proposed method employs physics-in-the-loop (PITL) optimization -- evaluating candidate solutions in actual environments to account for device specifications and disturbances -- and utilizes a distributed covariance matrix adaptation evolution strategy. Experiments confirmed that the proposed method successfully created adversarial examples that lead to depth misestimations, resulting in parts of objects disappearing from the target scene.

摘要: 深度神经网络（DNNs）在面对输入图像添加特定扰动时仍易受对抗攻击而导致误分类。这种脆弱性同样威胁着基于DNN的单目深度估计（MDE）模型的可靠性，使得在实际应用中增强鲁棒性成为关键需求。为验证基于DNN的MDE模型的脆弱性，本研究提出一种基于投影的对抗攻击方法，将扰动光线投射至目标物体上。该方法采用物理闭环（PITL）优化——通过在真实环境中评估候选解以考虑设备规格和干扰因素，并利用分布式协方差矩阵自适应进化策略。实验证实，所提方法成功生成了导致深度估计错误的对抗样本，使得目标场景中物体的部分区域消失。



## **4. Real-World Adversarial Attacks on RF-Based Drone Detectors**

针对基于射频的无人机探测器的现实世界对抗攻击 cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2512.20712v2) [paper-pdf](https://arxiv.org/pdf/2512.20712v2)

**Confidence**: 0.95

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.

摘要: 基于射频（RF）的系统通过分析无人机的射频信号模式，将其转换为频谱图图像，并由目标检测模型处理，从而用于无人机检测。现有的针对基于图像的模型的射频攻击会改变数字特征，但由于将数字扰动转换为可传输波形可能引入同步误差和干扰，并受硬件限制，使得空中（OTA）实施变得困难。我们首次提出了针对基于射频图像的无人机探测器的物理攻击，优化了特定类别的通用复基带（I/Q）扰动波形，这些波形与合法通信信号一同传输。我们使用射频记录和四种类型无人机的OTA实验评估了该攻击。结果表明，适度、结构化的I/Q扰动与标准射频链兼容，能可靠地降低目标无人机的检测率，同时保持对合法无人机的检测能力。



## **5. Misspecified Crame-Rao Bound for AoA Estimation at a ULA under a Spoofing Attack**

欺骗攻击下ULA AoA估计的误设定克拉美-罗界 eess.SP

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16735v1) [paper-pdf](https://arxiv.org/pdf/2512.16735v1)

**Confidence**: 0.95

**Authors**: Sotiris Skaperas, Arsenia Chorti

**Abstract**: A framework is presented for analyzing the impact of active attacks to location-based physical layer authentication (PLA) using the machinery of misspecified Cramér--Rao bound (MCRB). In this work, we focus on the MCRB in the angle-of-arrival (AoA) based authentication of a single antenna user when the verifier posseses an $M$ antenna element uniform linear array (ULA), assuming deterministic pilot signals; in our system model the presence of a spoofing adversary with an arbitrary number $L$ of antenna elements is assumed. We obtain a closed-form expression for the MCRB and demonstrate that the attack introduces in it a penalty term compared to the classic CRB, which does not depend on the signal-to-noise ratio (SNR) but on the adversary's location, the array geometry and the attacker precoding vector.

摘要: 本文提出一个利用误设定克拉美-罗界（MCRB）分析主动攻击对基于位置的物理层认证（PLA）影响的框架。本研究聚焦于单天线用户的角度到达（AoA）认证场景：验证方配备M阵元均匀线性阵列（ULA），采用确定性导频信号；系统模型中假设存在一个具有任意L个阵元的欺骗攻击者。我们推导出MCRB的闭式表达式，并证明相较于经典CRB，攻击会引入一个惩罚项——该惩罚项不依赖于信噪比（SNR），而取决于攻击者位置、阵列几何结构及攻击者预编码向量。



## **6. Talking to the Airgap: Exploiting Radio-Less Embedded Devices as Radio Receivers**

与空气隙对话：利用无射频嵌入式设备作为无线电接收器 cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15387v1) [paper-pdf](https://arxiv.org/pdf/2512.15387v1)

**Confidence**: 0.95

**Authors**: Paul Staat, Daniel Davidovich, Christof Paar

**Abstract**: Intelligent electronics are deeply embedded in critical infrastructures and must remain reliable, particularly against deliberate attacks. To minimize risks and impede remote compromise, sensitive systems can be physically isolated from external networks, forming an airgap. Yet, airgaps can still be infiltrated by capable adversaries gaining code execution. Prior research has shown that attackers can then attempt to wirelessly exfiltrate data across the airgap by exploiting unintended radio emissions. In this work, we demonstrate reversal of this link: malicious code execution on embedded devices can enable wireless infiltration of airgapped systems without any hardware modification. In contrast to previous infiltration methods that depend on dedicated sensors (e.g., microphones, LEDs, or temperature sensors) or require strict line-of-sight, we show that unmodified, sensor-less embedded devices can inadvertently act as radio receivers. This phenomenon stems from parasitic RF sensitivity in PCB traces and on-chip analog-to-digital converters (ADCs), allowing external transmissions to be received and decoded entirely in software.   Across twelve commercially available embedded devices and two custom prototypes, we observe repeatable reception in the 300-1000 MHz range, with detectable signal power as low as 1 mW. To this end, we propose a systematic methodology to identify device configurations that foster such radio sensitivities and comprehensively evaluate their feasibility for wireless data reception. Exploiting these sensitivities, we demonstrate successful data reception over tens of meters, even in non-line-of-sight conditions and show that the reception sensitivities accommodate data rates of up to 100 kbps. Our findings reveal a previously unexplored command-and-control vector for air-gapped systems while challenging assumptions about their inherent isolation. [shortened]

摘要: 智能电子设备已深度嵌入关键基础设施，必须保持可靠性，尤其要防范蓄意攻击。为降低风险并阻止远程入侵，敏感系统可采用物理隔离方式与外部网络断开连接，形成空气隙。然而，具备能力的攻击者仍可通过代码执行渗透空气隙系统。先前研究表明，攻击者可利用意外射频辐射实现跨空气隙的无线数据窃取。本研究则展示了逆向链路：通过在嵌入式设备上执行恶意代码，无需硬件改造即可实现对空气隙系统的无线渗透。与以往依赖专用传感器（如麦克风、LED或温度传感器）或严格视距条件的渗透方法不同，我们证明未经改造的无传感器嵌入式设备可能无意中成为无线电接收器。这种现象源于PCB走线和片上模数转换器（ADC）的寄生射频敏感性，使得外部传输信号可完全通过软件进行接收解码。通过对12款商用嵌入式设备和2款定制原型机的测试，我们在300-1000 MHz频段观察到可重复的接收现象，最低可检测信号功率达1 mW。为此，我们提出系统化方法以识别易产生此类射频敏感性的设备配置，并全面评估其无线数据接收可行性。利用这些敏感性，我们成功演示了数十米距离（包括非视距条件）的数据接收，接收灵敏度最高可支持100 kbps数据传输速率。这些发现揭示了空气隙系统先前未被探索的命令与控制向量，同时对其固有隔离性的假设提出了挑战。



## **7. PHANTOM: PHysical ANamorphic Threats Obstructing Connected Vehicle Mobility**

PHANTOM：阻碍网联车辆移动性的物理变形威胁 cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.19711v1) [paper-pdf](https://arxiv.org/pdf/2512.19711v1)

**Confidence**: 0.95

**Authors**: Md Nahid Hasan Shuvo, Moinul Hossain

**Abstract**: Connected autonomous vehicles (CAVs) rely on vision-based deep neural networks (DNNs) and low-latency (Vehicle-to-Everything) V2X communication to navigate safely and efficiently. Despite their advances, these systems remain vulnerable to physical adversarial attacks. In this paper, we introduce PHANTOM (PHysical ANamorphic Threats Obstructing connected vehicle Mobility), a novel framework for crafting and deploying perspective-dependent adversarial examples using \textit{anamorphic art}. PHANTOM exploits geometric distortions that appear natural to humans but are misclassified with high confidence by state-of-the-art object detectors. Unlike conventional attacks, PHANTOM operates in black-box settings without model access and demonstrates strong transferability across four diverse detector architectures (YOLOv5, SSD, Faster R-CNN, and RetinaNet). Comprehensive evaluation in CARLA across varying speeds, weather conditions, and lighting scenarios shows that PHANTOM achieves over 90\% attack success rate under optimal conditions and maintains 60-80\% effectiveness even in degraded environments. The attack activates within 6-10 meters of the target, providing insufficient time for safe maneuvering. Beyond individual vehicle deception, PHANTOM triggers network-wide disruption in CAV systems: SUMO-OMNeT++ co-simulation demonstrates that false emergency messages propagate through V2X links, increasing Peak Age of Information by 68-89\% and degrading safety-critical communication. These findings expose critical vulnerabilities in both perception and communication layers of CAV ecosystems.

摘要: 网联自动驾驶车辆（CAV）依赖基于视觉的深度神经网络（DNN）和低延迟车联网（V2X）通信来实现安全高效导航。尽管技术不断进步，这些系统仍易受物理对抗攻击。本文提出PHANTOM（阻碍网联车辆移动性的物理变形威胁）——一种利用变形艺术构建和部署视角依赖性对抗样本的新型框架。PHANTOM利用几何畸变，这些畸变对人类视觉呈现自然状态，却能以高置信度误导先进目标检测器。与传统攻击不同，PHANTOM在无需模型访问的黑盒环境下运行，并在四种异构检测器架构（YOLOv5、SSD、Faster R-CNN和RetinaNet）上展现出强迁移性。在CARLA仿真环境中进行的多维度评估（涵盖不同车速、天气条件和光照场景）表明：PHANTOM在最优条件下攻击成功率超过90%，即使在恶劣环境下仍保持60-80%的有效性。该攻击在目标6-10米范围内激活，使安全规避时间严重不足。除单车欺骗外，PHANTOM还会引发CAV系统级网络中断：SUMO-OMNeT++联合仿真显示，虚假紧急信息通过V2X链路传播，导致信息峰值时效性恶化68-89%，严重损害安全关键通信。这些发现揭示了CAV生态系统在感知层和通信层的双重关键脆弱性。



## **8. Exposing Vulnerabilities in Counterfeit Prevention Systems Utilizing Physically Unclonable Surface Features**

利用物理不可克隆表面特征的防伪系统漏洞暴露 cs.CR

15 pages; This work builds on arXiv:2408.02221 [cs.CR]

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.09150v1) [paper-pdf](https://arxiv.org/pdf/2512.09150v1)

**Confidence**: 0.95

**Authors**: Anirudh Nakra, Nayeeb Rashid, Chau-Wai Wong, Min Wu

**Abstract**: Counterfeit products pose significant risks to public health and safety through infiltrating untrusted supply chains. Among numerous anti-counterfeiting techniques, leveraging inherent, unclonable microscopic irregularities of paper surfaces is an accurate and cost-effective solution. Prior work of this approach has focused on enabling ubiquitous acquisition of these physically unclonable features (PUFs). However, we will show that existing authentication methods relying on paper surface PUFs may be vulnerable to adversaries, resulting in a gap between technological feasibility and secure real-world deployment. This gap is investigated through formalizing an operational framework for paper-PUF-based authentication. Informed by this framework, we reveal system-level vulnerabilities across both physical and digital domains, designing physical denial-of-service and digital forgery attacks to disrupt proper authentication. The effectiveness of the designed attacks underscores the strong need for security countermeasures for reliable and resilient authentication based on paper PUFs. The proposed framework further facilitates a comprehensive, stage-by-stage security analysis, guiding the design of future counterfeit prevention systems. This analysis delves into potential attack strategies, offering a foundational understanding of how various system components, such as physical features and verification processes, might be exploited by adversaries.

摘要: 假冒产品通过渗透不可信供应链对公共健康与安全构成重大风险。在众多防伪技术中，利用纸张表面固有的、不可克隆的微观不规则性是一种准确且经济高效的解决方案。该方法先前的研究侧重于实现对这些物理不可克隆特征（PUFs）的普适采集。然而，我们将证明依赖纸张表面PUFs的现有认证方法可能易受攻击者利用，导致技术可行性与安全实际部署之间存在差距。通过形式化基于纸张PUF认证的操作框架，我们研究了这一差距。基于该框架，我们揭示了跨越物理和数字领域的系统级漏洞，设计了物理拒绝服务和数字伪造攻击以干扰正常认证。所设计攻击的有效性凸显了对基于纸张PUFs的可靠、弹性认证实施安全对策的迫切需求。所提出的框架进一步促进了分阶段、全面的安全分析，指导未来防伪系统的设计。该分析深入探讨了潜在攻击策略，为理解物理特征、验证流程等各种系统组件如何可能被攻击者利用提供了基础认知。



## **9. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

物理逼真的序列级对抗性服装：实现鲁棒的人体检测规避 cs.CV

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.16020v2) [paper-pdf](https://arxiv.org/pdf/2511.16020v2)

**Confidence**: 0.95

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.

摘要: 用于人体检测的深度神经网络极易受到对抗性操纵，在真实监控环境中带来安全和隐私风险。可穿戴攻击提供了一种现实的威胁模型，但现有方法通常逐帧优化纹理，因此无法在包含运动、姿态变化和衣物变形的长视频序列中保持隐蔽性。本研究引入序列级优化框架，为衬衫、裤子和帽子生成自然、可打印的对抗性纹理，这些纹理在数字和物理环境下的整个行走视频中持续有效。首先将产品图像映射到UV空间，并转换为紧凑的调色板和控制点参数化，通过ICC锁定确保所有颜色可打印。随后采用基于物理的人体-服装流程模拟运动、多角度摄像机视角、布料动力学和光照变化。使用时序加权的变换期望目标优化控制点，使整个序列中的检测置信度最小化。大量实验证明了强大而稳定的隐蔽性、对视角变化的高鲁棒性以及卓越的跨模型可迁移性。通过升华印花制作的物理服装在室内外录制中实现了可靠的检测抑制，证实了现实世界的可行性。



## **10. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

面向车辆检测器的鲁棒准确对抗伪装生成方法 cs.CV

14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2411.10029v2) [paper-pdf](https://arxiv.org/pdf/2411.10029v2)

**Confidence**: 0.95

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装因其在多视角攻击性能上的优势，已成为针对车辆检测器广泛使用的物理攻击手段。一种有前景的方法是利用可微分神经渲染器，通过梯度反向传播促进对抗伪装优化。然而，现有方法在渲染过程中往往难以捕捉环境特征，或生成的对抗纹理无法精确映射到目标车辆。此外，这些方法忽略了多样化的天气条件，导致生成的伪装在不同天气场景下的有效性降低。为应对这些挑战，我们提出了一种鲁棒且准确的伪装生成方法RAUCA。其核心是一个新颖的神经渲染组件——端到端神经渲染器增强版（E2E-NRP），能够准确优化和投射车辆纹理，并渲染包含光照、天气等环境特征的图像。此外，我们整合了多天气数据集用于伪装生成，利用E2E-NRP增强攻击鲁棒性。在六种主流目标检测器上的实验结果表明，RAUCA-final在仿真和真实场景中均优于现有方法。



## **11. Visual Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗攻击与防御：综述 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2211.01671v6) [paper-pdf](https://arxiv.org/pdf/2211.01671v6)

**Confidence**: 0.95

**Authors**: Xingxing Wei, Bangzheng Pu, Shiji Zhao, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they remain vulnerable to adversarial examples. Adversarial attacks in computer vision can be categorized into digital attacks and physical attacks based on their different forms. Compared to digital attacks, which generate perturbations in digital pixels, physical attacks are more practical in real-world settings. Due to the serious security risks posed by physically adversarial examples, many studies have been conducted to evaluate the physically adversarial robustness of DNNs in recent years. In this paper, we provide a comprehensive survey of current physically adversarial attacks and defenses in computer vision. We establish a taxonomy by organizing physical attacks according to attack tasks, attack forms, and attack methods. This approach offers readers a systematic understanding of the topic from multiple perspectives. For physical defenses, we categorize them into pre-processing, in-processing, and post-processing for DNN models to ensure comprehensive coverage of adversarial defenses. Based on this survey, we discuss the challenges facing this research field and provide an outlook on future directions.

摘要: 尽管深度神经网络（DNNs）已在多种现实场景中得到广泛应用，但它们仍易受对抗样本的影响。计算机视觉中的对抗攻击根据其形式可分为数字攻击和物理攻击。与在数字像素中生成扰动的数字攻击相比，物理攻击在现实环境中更具实用性。由于物理对抗样本带来的严重安全风险，近年来已有大量研究致力于评估DNNs的物理对抗鲁棒性。本文全面综述了当前计算机视觉领域的物理对抗攻击与防御方法。我们通过按攻击任务、攻击形式和攻击方法对物理攻击进行分类，建立了一个分类体系，从而为读者提供多视角的系统性理解。针对物理防御，我们将其分为DNN模型的前处理、处理中和后处理三类，以确保全面覆盖对抗防御策略。基于本综述，我们讨论了该研究领域面临的挑战，并对未来发展方向进行了展望。



## **12. Physical ID-Transfer Attacks against Multi-Object Tracking via Adversarial Trajectory**

基于对抗轨迹的多目标跟踪物理ID转移攻击 cs.CV

Accepted to Annual Computer Security Applications Conference (ACSAC) 2024

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01934v1) [paper-pdf](https://arxiv.org/pdf/2512.01934v1)

**Confidence**: 0.95

**Authors**: Chenyi Wang, Yanmao Man, Raymond Muller, Ming Li, Z. Berkay Celik, Ryan Gerdes, Jonathan Petit

**Abstract**: Multi-Object Tracking (MOT) is a critical task in computer vision, with applications ranging from surveillance systems to autonomous driving. However, threats to MOT algorithms have yet been widely studied. In particular, incorrect association between the tracked objects and their assigned IDs can lead to severe consequences, such as wrong trajectory predictions. Previous attacks against MOT either focused on hijacking the trackers of individual objects, or manipulating the tracker IDs in MOT by attacking the integrated object detection (OD) module in the digital domain, which are model-specific, non-robust, and only able to affect specific samples in offline datasets. In this paper, we present AdvTraj, the first online and physical ID-manipulation attack against tracking-by-detection MOT, in which an attacker uses adversarial trajectories to transfer its ID to a targeted object to confuse the tracking system, without attacking OD. Our simulation results in CARLA show that AdvTraj can fool ID assignments with 100% success rate in various scenarios for white-box attacks against SORT, which also have high attack transferability (up to 93% attack success rate) against state-of-the-art (SOTA) MOT algorithms due to their common design principles. We characterize the patterns of trajectories generated by AdvTraj and propose two universal adversarial maneuvers that can be performed by a human walker/driver in daily scenarios. Our work reveals under-explored weaknesses in the object association phase of SOTA MOT systems, and provides insights into enhancing the robustness of such systems.

摘要: 多目标跟踪（MOT）是计算机视觉中的关键任务，广泛应用于监控系统和自动驾驶等领域。然而，针对MOT算法的威胁尚未得到广泛研究。特别是，跟踪对象与其分配ID之间的错误关联可能导致严重后果，例如错误的轨迹预测。以往针对MOT的攻击要么侧重于劫持单个对象的跟踪器，要么通过攻击数字域中的集成目标检测（OD）模块来操纵MOT中的跟踪器ID，这些攻击具有模型特定性、非鲁棒性，且仅能影响离线数据集中的特定样本。本文提出AdvTraj，这是首个针对检测跟踪式MOT的在线物理ID操纵攻击，攻击者利用对抗轨迹将其ID转移至目标对象以混淆跟踪系统，而无需攻击OD。在CARLA中的仿真结果表明，AdvTraj在白盒攻击SORT时，在各种场景下能以100%的成功率欺骗ID分配，并且由于先进MOT算法的共同设计原则，该攻击对SOTA MOT算法具有高攻击可迁移性（攻击成功率高达93%）。我们分析了AdvTraj生成的轨迹模式，并提出了两种可由人类步行者/驾驶员在日常场景中执行的通用对抗机动策略。本研究揭示了SOTA MOT系统在对象关联阶段未被充分探索的弱点，并为增强此类系统的鲁棒性提供了见解。



## **13. The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches**

欺骗轮廓：利用边缘补丁对交通标志实施物理对抗攻击 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.00765v2) [paper-pdf](https://arxiv.org/pdf/2512.00765v2)

**Confidence**: 0.95

**Authors**: Haojie Ji, Te Hu, Haowen Li, Long Jin, Chongshi Xin, Yuchi Yao, Jiarui Xiao

**Abstract**: Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.

摘要: 智能驾驶系统易受针对交通标志的物理对抗攻击。此类攻击可导致分类错误，引发危及道路安全的驾驶决策失误。在V2X网络中，这类误判可能进一步传播，引发级联故障，破坏整体交通流与系统稳定性。然而，现有物理攻击方法普遍存在隐蔽性不足的缺陷：多数方法在标志中心区域施加扰动，产生视觉上显著的异常图案，易被人类观察者察觉，限制了实际应用价值。本研究提出TESP-Attack——一种面向交通标志分类的新型隐蔽感知对抗补丁方法。基于人类视觉注意力主要集中于交通标志中心区域的观察，我们采用实例分割技术生成符合标志形状特征的边缘对齐掩码。通过U-Net生成器构建对抗补丁，并借助色彩纹理约束与频域分析进行优化，实现与背景环境的无缝融合，达成高度视觉隐蔽效果。该方法在多种架构的交通标志分类模型上均展现出卓越的攻击成功率，在有限查询预算下超过90%，同时具备强大的跨模型迁移能力，并在不同角度与距离条件下保持稳定的现实场景攻击性能。



## **14. Adversarial Patch Attacks on Vision-Based Cargo Occupancy Estimation via Differentiable 3D Simulation**

基于可微分3D模拟的视觉货物占用率估计对抗性补丁攻击 cs.CV

9 pages, 5 figures, 1 algorithm

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19254v1) [paper-pdf](https://arxiv.org/pdf/2511.19254v1)

**Confidence**: 0.95

**Authors**: Mohamed Rissal Hedna, Sesugh Samuel Nder

**Abstract**: Computer vision systems are increasingly adopted in modern logistics operations, including the estimation of trailer occupancy for planning, routing, and billing. Although effective, such systems may be vulnerable to physical adversarial attacks, particularly adversarial patches that can be printed and placed on interior surfaces. In this work, we study the feasibility of such attacks on a convolutional cargo-occupancy classifier using fully simulated 3D environments. Using Mitsuba 3 for differentiable rendering, we optimize patch textures across variations in geometry, lighting, and viewpoint, and compare their effectiveness to a 2D compositing baseline. Our experiments demonstrate that 3D-optimized patches achieve high attack success rates, especially in a denial-of-service scenario (empty to full), where success reaches 84.94 percent. Concealment attacks (full to empty) prove more challenging but still reach 30.32 percent. We analyze the factors influencing attack success, discuss implications for the security of automated logistics pipelines, and highlight directions for strengthening physical robustness. To our knowledge, this is the first study to investigate adversarial patch attacks for cargo-occupancy estimation in physically realistic, fully simulated 3D scenes.

摘要: 计算机视觉系统在现代物流作业中的应用日益广泛，包括用于规划、路线安排和计费的拖车占用率估计。尽管这些系统有效，但它们可能容易受到物理对抗性攻击，特别是可打印并放置在内部表面的对抗性补丁。在本研究中，我们利用完全模拟的3D环境，探讨了此类攻击对卷积货物占用率分类器的可行性。通过使用Mitsuba 3进行可微分渲染，我们在几何、光照和视点变化中优化补丁纹理，并将其有效性与2D合成基线进行比较。实验表明，3D优化补丁实现了较高的攻击成功率，尤其在拒绝服务场景（空到满）中，成功率可达84.94%。隐藏攻击（满到空）更具挑战性，但仍达到30.32%。我们分析了影响攻击成功的因素，讨论了自动化物流管道安全性的影响，并指出了增强物理鲁棒性的方向。据我们所知，这是首个在物理真实、完全模拟的3D场景中研究货物占用率估计对抗性补丁攻击的工作。



## **15. Robust Physical Adversarial Patches Using Dynamically Optimized Clusters**

基于动态优化聚类的鲁棒物理对抗性补丁 cs.CV

Supplementary material available at: https://drive.google.com/drive/folders/1Yntcc9CARdbvoJJ51cyUm1DWGSvU9X4V?usp=drive_link

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18656v1) [paper-pdf](https://arxiv.org/pdf/2511.18656v1)

**Confidence**: 0.95

**Authors**: Harrison Bagley, Will Meakin, Simon Lucey, Yee Wei Law, Tat-Jun Chin

**Abstract**: Physical adversarial attacks on deep learning systems is concerning due to the ease of deploying such attacks, usually by placing an adversarial patch in a scene to manipulate the outcomes of a deep learning model. Training such patches typically requires regularization that improves physical realizability (e.g., printability, smoothness) and/or robustness to real-world variability (e.g. deformations, viewing angle, noise). One type of variability that has received little attention is scale variability. When a patch is rescaled, either digitally through downsampling/upsampling or physically through changing imaging distances, interpolation-induced color mixing occurs. This smooths out pixel values, resulting in a loss of high-frequency patterns and degrading the adversarial signal. To address this, we present a novel superpixel-based regularization method that guides patch optimization to scale-resilient structures. Our ap proach employs the Simple Linear Iterative Clustering (SLIC) algorithm to dynamically cluster pixels in an adversarial patch during optimization. The Implicit Function Theorem is used to backpropagate gradients through SLIC to update the superpixel boundaries and color. This produces patches that maintain their structure over scale and are less susceptible to interpolation losses. Our method achieves greater performance in the digital domain, and when realized physically, these performance gains are preserved, leading to improved physical performance. Real-world performance was objectively assessed using a novel physical evaluation protocol that utilizes screens and cardboard cut-outs to systematically vary real-world conditions.

摘要: 针对深度学习系统的物理对抗性攻击令人担忧，因为此类攻击易于部署，通常通过在场景中放置对抗性补丁来操纵深度学习模型的输出。训练此类补丁通常需要正则化，以提高物理可实现性（例如，可打印性、平滑度）和/或对现实世界变化的鲁棒性（例如，变形、视角、噪声）。其中，尺度变化性受到的关注较少。当补丁被重新缩放时，无论是通过数字下采样/上采样还是通过改变成像距离物理缩放，都会发生插值引起的颜色混合。这会平滑像素值，导致高频模式丢失并削弱对抗性信号。为解决此问题，我们提出了一种新颖的基于超像素的正则化方法，引导补丁优化形成对尺度变化具有弹性的结构。我们的方法采用简单线性迭代聚类（SLIC）算法，在优化过程中动态聚类对抗性补丁中的像素。利用隐函数定理通过SLIC反向传播梯度，以更新超像素边界和颜色。这产生的补丁在尺度变化下能保持其结构，且不易受插值损失影响。我们的方法在数字域中实现了更优性能，当物理实现时，这些性能优势得以保留，从而提升了物理攻击效果。我们使用一种新颖的物理评估协议客观评估了现实世界性能，该协议利用屏幕和纸板剪裁来系统性地改变现实世界条件。



## **16. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

全尺度欺骗立体匹配：针对自动驾驶中双目深度估计的物理对抗攻击 cs.CV

AAAI 2026

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.14386v3) [paper-pdf](https://arxiv.org/pdf/2511.14386v3)

**Confidence**: 0.95

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

摘要: 尽管用于实现自动驾驶感知的深度神经网络模型已被证明易受对抗样本攻击，但已知攻击通常利用2D补丁并主要针对单目感知。因此，物理对抗样本（PAEs）对基于立体的双目深度估计的有效性在很大程度上仍未得到探索。为此，我们提出了首个针对自动驾驶场景中立体匹配模型的纹理化物理对抗攻击方法。我们的方法采用具有全局伪装纹理的3D PAE，而非基于局部2D补丁的攻击，确保在立体相机不同视角下既保持视觉一致性又具备攻击效果。为应对这些相机的视差效应，我们还提出了一种新的3D立体匹配渲染模块，使PAE能够在双目视觉中与现实世界的位置和朝向对齐。我们进一步提出了一种新颖的融合攻击，通过细粒度PAE优化将目标无缝融入环境。相较于现有无法与背景无缝融合的隐藏攻击，该方法显著增强了隐蔽性和杀伤力。大量评估表明，我们的PAEs能够成功欺骗立体模型产生错误的深度信息。



## **17. Post-Quantum Cryptography for Intelligent Transportation Systems: An Implementation-Focused Review**

智能交通系统的后量子密码学：以实施为重点的综述 cs.CR

This is a preprint version of a manuscript currently under peer review. This version has not undergone peer review and may differ from the final published version

**SubmitDate**: 2026-01-03    [abs](http://arxiv.org/abs/2601.01068v1) [paper-pdf](https://arxiv.org/pdf/2601.01068v1)

**Confidence**: 0.85

**Authors**: Abdullah Al Mamun, Akid Abrar, Mizanur Rahman, M Sabbir Salek, Mashrur Chowdhury

**Abstract**: As quantum computing advances, the cryptographic algorithms that underpin confidentiality, integrity, and authentication in Intelligent Transportation Systems (ITS) face increasing vulnerability to quantum-enabled attacks. To address these risks, governments and industry stakeholders are turning toward post-quantum cryptography (PQC), a class of algorithms designed to resist adversaries equipped with quantum computing capabilities. However, existing studies provide limited insight into the implementation-focused aspects of PQC in the ITS domain. This review fills that gap by evaluating the readiness of vehicular communication and security standards for PQC adoption. It examines in-vehicle networks and vehicle-to-everything (V2X) interfaces, while also investigating vulnerabilities at the physical layer, primarily exposure to side-channel and fault injection attacks. The review identifies thirteen research gaps reflecting non-PQC-ready standards, constraints in embedded implementation and hybrid cryptography, interoperability and certificate-management barriers, lack of real-world PQC deployment data in ITS, and physical-attack vulnerabilities in PQC-enabled vehicular communication. Future research directions include updating vehicular communication and security standards, optimizing PQC for low-power devices, enhancing interoperability and certificate-management frameworks for PQC integration, conducting real-world evaluations of PQC-enabled communication and control functions across ITS deployments, and strengthening defenses against AI-assisted physical attacks. A phased roadmap is presented, aligning PQC deployment with regulatory, performance, and safety requirements, thereby guiding the secure evolution of ITS in the quantum computing era.

摘要: 随着量子计算的发展，支撑智能交通系统（ITS）中机密性、完整性和认证的密码算法面临日益增长的量子攻击威胁。为应对这些风险，各国政府和行业利益相关者正转向后量子密码学（PQC）——一类旨在抵御具备量子计算能力的攻击者的算法。然而，现有研究对PQC在ITS领域以实施为重点的方面提供有限见解。本综述通过评估车载通信和安全标准对PQC采用的准备情况来填补这一空白。它考察了车内网络和车联万物（V2X）接口，同时调查了物理层的脆弱性，主要是侧信道和故障注入攻击的暴露风险。本综述识别了十三个研究缺口，反映了标准未做好PQC准备、嵌入式实施和混合密码学的约束、互操作性和证书管理障碍、ITS中缺乏真实世界PQC部署数据，以及启用PQC的车载通信中的物理攻击脆弱性。未来研究方向包括：更新车载通信和安全标准、为低功耗设备优化PQC、增强PQC集成的互操作性和证书管理框架、在ITS部署中对启用PQC的通信和控制功能进行真实世界评估，以及加强针对AI辅助物理攻击的防御。本文提出了一个分阶段路线图，将PQC部署与监管、性能和安全要求对齐，从而指导量子计算时代ITS的安全演进。



