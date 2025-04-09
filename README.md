# Latest Adversarial Attack Papers
**update at 2025-04-09 10:27:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

cs.RO

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06154v1) [paper-pdf](http://arxiv.org/pdf/2504.06154v1)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.



## **2. Frequency maps reveal the correlation between Adversarial Attacks and Implicit Bias**

cs.LG

Accepted at IJCNN 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2305.15203v3) [paper-pdf](http://arxiv.org/pdf/2305.15203v3)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto d'Onofrio, Luca Manzoni, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the correlation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse a representation of the network's implicit bias through the lens of the Fourier transform. Specifically, we identify unique fingerprints of implicit bias and adversarial attacks by calculating the minimal, essential frequencies needed for accurate classification of each image, as well as the frequencies that drive misclassification in its adversarially perturbed counterpart. This approach enables us to uncover and analyse the correlation between these essential frequencies, providing a precise map of how the network's biases align or contrast with the frequency components exploited by adversarial attacks. To this end, among other methods, we use a newly introduced technique capable of detecting nonlinear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.



## **3. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking**

cs.CV

Accepted by CVPR2025 as Highlight

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05838v1) [paper-pdf](http://arxiv.org/pdf/2504.05838v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, the Image Prompt Adapter (IP-Adapter) has been increasingly integrated into text-to-image diffusion models (T2I-DMs) to improve controllability. However, in this paper, we reveal that T2I-DMs equipped with the IP-Adapter (T2I-IP-DMs) enable a new jailbreak attack named the hijacking attack. We demonstrate that, by uploading imperceptible image-space adversarial examples (AEs), the adversary can hijack massive benign users to jailbreak an Image Generation Service (IGS) driven by T2I-IP-DMs and mislead the public to discredit the service provider. Worse still, the IP-Adapter's dependency on open-source image encoders reduces the knowledge required to craft AEs. Extensive experiments verify the technical feasibility of the hijacking attack. In light of the revealed threat, we investigate several existing defenses and explore combining the IP-Adapter with adversarially trained models to overcome existing defenses' limitations. Our code is available at https://github.com/fhdnskfbeuv/attackIPA.



## **4. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.



## **5. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

Accepted to FSE 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2410.22663v2) [paper-pdf](http://arxiv.org/pdf/2410.22663v2)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.



## **6. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

eess.AS

This manuscript has been submitted for peer review

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05657v1) [paper-pdf](http://arxiv.org/pdf/2504.05657v1)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.



## **7. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.



## **8. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.



## **9. ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs**

cs.CR

Zhao et al., 16 pages, 2025, uploaded by Hanzhou Wu, Shanghai  University

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05605v1) [paper-pdf](http://arxiv.org/pdf/2504.05605v1)

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang, Athanasios V. Vasilakos

**Abstract**: Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency.



## **10. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05592v1) [paper-pdf](http://arxiv.org/pdf/2504.05592v1)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBR) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetrations through Hardware-in-the-Loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.



## **11. Secure Diagnostics: Adversarial Robustness Meets Clinical Interpretability**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05483v1) [paper-pdf](http://arxiv.org/pdf/2504.05483v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadreza Pashanejad, Saman Soleimani Roudi, Mohammad Norouzi, Saeed Bagheri Shouraki

**Abstract**: Deep neural networks for medical image classification often fail to generalize consistently in clinical practice due to violations of the i.i.d. assumption and opaque decision-making. This paper examines interpretability in deep neural networks fine-tuned for fracture detection by evaluating model performance against adversarial attack and comparing interpretability methods to fracture regions annotated by an orthopedic surgeon. Our findings prove that robust models yield explanations more aligned with clinically meaningful areas, indicating that robustness encourages anatomically relevant feature prioritization. We emphasize the value of interpretability for facilitating human-AI collaboration, in which models serve as assistants under a human-in-the-loop paradigm: clinically plausible explanations foster trust, enable error correction, and discourage reliance on AI for high-stakes decisions. This paper investigates robustness and interpretability as complementary benchmarks for bridging the gap between benchmark performance and safe, actionable clinical deployment.



## **12. Adversarial KA**

cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05255v1) [paper-pdf](http://arxiv.org/pdf/2504.05255v1)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.



## **13. Security Risks in Vision-Based Beam Prediction: From Spatial Proxy Attacks to Feature Refinement**

cs.NI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05222v1) [paper-pdf](http://arxiv.org/pdf/2504.05222v1)

**Authors**: Avi Deb Raha, Kitae Kim, Mrityunjoy Gain, Apurba Adhikary, Zhu Han, Eui-Nam Huh, Choong Seon Hong

**Abstract**: The rapid evolution towards the sixth-generation (6G) networks demands advanced beamforming techniques to address challenges in dynamic, high-mobility scenarios, such as vehicular communications. Vision-based beam prediction utilizing RGB camera images emerges as a promising solution for accurate and responsive beam selection. However, reliance on visual data introduces unique vulnerabilities, particularly susceptibility to adversarial attacks, thus potentially compromising beam accuracy and overall network reliability. In this paper, we conduct the first systematic exploration of adversarial threats specifically targeting vision-based mmWave beam selection systems. Traditional white-box attacks are impractical in this context because ground-truth beam indices are inaccessible and spatial dynamics are complex. To address this, we propose a novel black-box adversarial attack strategy, termed Spatial Proxy Attack (SPA), which leverages spatial correlations between user positions and beam indices to craft effective perturbations without requiring access to model parameters or labels. To counteract these adversarial vulnerabilities, we formulate an optimization framework aimed at simultaneously enhancing beam selection accuracy under clean conditions and robustness against adversarial perturbations. We introduce a hybrid deep learning architecture integrated with a dedicated Feature Refinement Module (FRM), designed to systematically filter irrelevant, noisy and adversarially perturbed visual features. Evaluations using standard backbone models such as ResNet-50 and MobileNetV2 demonstrate that our proposed method significantly improves performance, achieving up to an +21.07\% gain in Top-K accuracy under clean conditions and a 41.31\% increase in Top-1 adversarial robustness compared to different baseline models.



## **14. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Models**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.01440v3) [paper-pdf](http://arxiv.org/pdf/2412.01440v3)

**Authors**: Zhixiang Wang, Xiaosen Wang, Bo Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose DiffPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using DiffPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.



## **15. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.20006v3) [paper-pdf](http://arxiv.org/pdf/2412.20006v3)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Rapidly growing wildfires have recently devastated societal assets, exposing a critical need for early warning systems to expedite relief efforts. Smoke detection using camera-based Deep Neural Networks (DNNs) offers a promising solution for wildfire prediction. However, the rarity of smoke across time and space limits training data, raising model overfitting and bias concerns. Current DNNs, primarily Convolutional Neural Networks (CNNs) and transformers, complicate robustness evaluation due to architectural differences. To address these challenges, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating wildfire detection models' adversarial robustness. WARP addresses inherent limitations in data diversity by generating adversarial examples through image-global and -local perturbations. Global and local attacks superimpose Gaussian noise and PNG patches onto image inputs, respectively; this suits both CNNs and transformers while generating realistic adversarial scenarios. Using WARP, we assessed real-time CNNs and Transformers, uncovering key vulnerabilities. At times, transformers exhibited over 70% precision degradation under global attacks, while both models generally struggled to differentiate cloud-like PNG patches from real smoke during local attacks. To enhance model robustness, we proposed four wildfire-oriented data augmentation techniques based on WARP's methodology and results, which diversify smoke image data and improve model precision and robustness. These advancements represent a substantial step toward developing a reliable early wildfire warning system, which may be our first safeguard against wildfire destruction.



## **16. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **17. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.



## **18. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.16392v2) [paper-pdf](http://arxiv.org/pdf/2503.16392v2)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.



## **19. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **20. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

cs.CV

This paper has been accepted by ICME 2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.11408v2) [paper-pdf](http://arxiv.org/pdf/2408.11408v2)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.



## **21. Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios**

cs.RO

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.23708v2) [paper-pdf](http://arxiv.org/pdf/2503.23708v2)

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment.



## **22. Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models**

cs.CV

Accepted to CVPR2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04747v1) [paper-pdf](http://arxiv.org/pdf/2504.04747v1)

**Authors**: Yoojin Jung, Byung Cheol Song

**Abstract**: Deep learning-based computer vision systems adopt complex and large architectures to improve performance, yet they face challenges in deployment on resource-constrained mobile and edge devices. To address this issue, model compression techniques such as pruning, quantization, and matrix factorization have been proposed; however, these compressed models are often highly vulnerable to adversarial attacks. We introduce the \textbf{Efficient Ensemble Defense (EED)} technique, which diversifies the compression of a single base model based on different pruning importance scores and enhances ensemble diversity to achieve high adversarial robustness and resource efficiency. EED dynamically determines the number of necessary sub-models during the inference stage, minimizing unnecessary computations while maintaining high robustness. On the CIFAR-10 and SVHN datasets, EED demonstrated state-of-the-art robustness performance compared to existing adversarial pruning techniques, along with an inference speed improvement of up to 1.86 times. This proves that EED is a powerful defense solution in resource-constrained environments.



## **23. A Survey and Evaluation of Adversarial Attacks for Object Detection**

cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.01934v4) [paper-pdf](http://arxiv.org/pdf/2408.01934v4)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.



## **24. On the Robustness of GUI Grounding Models Against Image Attacks**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04716v1) [paper-pdf](http://arxiv.org/pdf/2504.04716v1)

**Authors**: Haoren Zhao, Tianyi Chen, Zhen Wang

**Abstract**: Graphical User Interface (GUI) grounding models are crucial for enabling intelligent agents to understand and interact with complex visual interfaces. However, these models face significant robustness challenges in real-world scenarios due to natural noise and adversarial perturbations, and their robustness remains underexplored. In this study, we systematically evaluate the robustness of state-of-the-art GUI grounding models, such as UGround, under three conditions: natural noise, untargeted adversarial attacks, and targeted adversarial attacks. Our experiments, which were conducted across a wide range of GUI environments, including mobile, desktop, and web interfaces, have clearly demonstrated that GUI grounding models exhibit a high degree of sensitivity to adversarial perturbations and low-resolution conditions. These findings provide valuable insights into the vulnerabilities of GUI grounding models and establish a strong benchmark for future research aimed at enhancing their robustness in practical applications. Our code is available at https://github.com/ZZZhr-1/Robust_GUI_Grounding.



## **25. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



## **26. Systematic Literature Review on Vehicular Collaborative Perception -- A Computer Vision Perspective**

cs.CV

39 pages, 25 figures

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04631v1) [paper-pdf](http://arxiv.org/pdf/2504.04631v1)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.



## **27. Selective Masking Adversarial Attack on Automatic Speech Recognition Systems**

cs.CR

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04394v1) [paper-pdf](http://arxiv.org/pdf/2504.04394v1)

**Authors**: Zheng Fang, Shenyi Zhang, Tao Wang, Bowen Li, Lingchen Zhao, Zhangyi Wang

**Abstract**: Extensive research has shown that Automatic Speech Recognition (ASR) systems are vulnerable to audio adversarial attacks. Current attacks mainly focus on single-source scenarios, ignoring dual-source scenarios where two people are speaking simultaneously. To bridge the gap, we propose a Selective Masking Adversarial attack, namely SMA attack, which ensures that one audio source is selected for recognition while the other audio source is muted in dual-source scenarios. To better adapt to the dual-source scenario, our SMA attack constructs the normal dual-source audio from the muted audio and selected audio. SMA attack initializes the adversarial perturbation with a small Gaussian noise and iteratively optimizes it using a selective masking optimization algorithm. Extensive experiments demonstrate that the SMA attack can generate effective and imperceptible audio adversarial examples in the dual-source scenario, achieving an average success rate of attack of 100% and signal-to-noise ratio of 37.15dB on Conformer-CTC, outperforming the baselines.



## **28. WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in Federated Learning for Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04367v1) [paper-pdf](http://arxiv.org/pdf/2504.04367v1)

**Authors**: Sameera K. M., Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the era of data expansion, ensuring data privacy has become increasingly critical, posing significant challenges to traditional AI-based applications. In addition, the increasing adoption of IoT devices has introduced significant cybersecurity challenges, making traditional Network Intrusion Detection Systems (NIDS) less effective against evolving threats, and privacy concerns and regulatory restrictions limit their deployment. Federated Learning (FL) has emerged as a promising solution, allowing decentralized model training while maintaining data privacy to solve these issues. However, despite implementing privacy-preserving technologies, FL systems remain vulnerable to adversarial attacks. Furthermore, data distribution among clients is not heterogeneous in the FL scenario. We propose WeiDetect, a two-phase, server-side defense mechanism for FL-based NIDS that detects malicious participants to address these challenges. In the first phase, local models are evaluated using a validation dataset to generate validation scores. These scores are then analyzed using a Weibull distribution, identifying and removing malicious models. We conducted experiments to evaluate the effectiveness of our approach in diverse attack settings. Our evaluation included two popular datasets, CIC-Darknet2020 and CSE-CIC-IDS2018, tested under non-IID data distributions. Our findings highlight that WeiDetect outperforms state-of-the-art defense approaches, improving higher target class recall up to 70% and enhancing the global model's F1 score by 1% to 14%.



## **29. Impact of Error Rate Misreporting on Resource Allocation in Multi-tenant Quantum Computing and Defense**

quant-ph

7 pages, 5 figures, conference

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04285v1) [paper-pdf](http://arxiv.org/pdf/2504.04285v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: Cloud-based quantum service providers allow multiple users to run programs on shared hardware concurrently to maximize resource utilization and minimize operational costs. This multi-tenant computing (MTC) model relies on the error parameters of the hardware for fair qubit allocation and scheduling, as error-prone qubits can degrade computational accuracy asymmetrically for users sharing the hardware. To maintain low error rates, quantum providers perform periodic hardware calibration, often relying on third-party calibration services. If an adversary within this calibration service misreports error rates, the allocator can be misled into making suboptimal decisions even when the physical hardware remains unchanged. We demonstrate such an attack model in which an adversary strategically misreports qubit error rates to reduce hardware throughput, and probability of successful trial (PST) for two previously proposed allocation frameworks, i.e. Greedy and Community-Based Dynamic Allocation Partitioning (COMDAP). Experimental results show that adversarial misreporting increases execution latency by 24% and reduces PST by 7.8%. We also propose to identify inconsistencies in reported error rates by analyzing statistical deviations in error rates across calibration cycles.



## **30. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2503.09334v2) [paper-pdf](http://arxiv.org/pdf/2503.09334v2)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.



## **31. Beating Backdoor Attack at Its Own Game**

cs.LG

Accepted to ICCV 2023

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2307.15539v4) [paper-pdf](http://arxiv.org/pdf/2307.15539v4)

**Authors**: Min Liu, Alberto Sangiovanni-Vincentelli, Xiangyu Yue

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attack, which does not affect the network's performance on clean data but would manipulate the network behavior once a trigger pattern is added. Existing defense methods have greatly reduced attack success rate, but their prediction accuracy on clean data still lags behind a clean model by a large margin. Inspired by the stealthiness and effectiveness of backdoor attack, we propose a simple but highly effective defense framework which injects non-adversarial backdoors targeting poisoned samples. Following the general steps in backdoor attack, we detect a small set of suspected samples and then apply a poisoning strategy to them. The non-adversarial backdoor, once triggered, suppresses the attacker's backdoor on poisoned data, but has limited influence on clean data. The defense can be carried out during data preprocessing, without any modification to the standard end-to-end training pipeline. We conduct extensive experiments on multiple benchmarks with different architectures and representative attacks. Results demonstrate that our method achieves state-of-the-art defense effectiveness with by far the lowest performance drop on clean data. Considering the surprising defense ability displayed by our framework, we call for more attention to utilizing backdoor for backdoor defense. Code is available at https://github.com/minliu01/non-adversarial_backdoor.



## **32. Authenticated Sublinear Quantum Private Information Retrieval**

quant-ph

11 pages, 1 figure

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04041v1) [paper-pdf](http://arxiv.org/pdf/2504.04041v1)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.



## **33. Disparate Privacy Vulnerability: Targeted Attribute Inference Attacks and Defenses**

cs.LG

Selected for publication at 34th USENIX Security Symposium

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04033v1) [paper-pdf](http://arxiv.org/pdf/2504.04033v1)

**Authors**: Ehsanul Kabir, Lucas Craig, Shagufta Mehnaz

**Abstract**: As machine learning (ML) technologies become more prevalent in privacy-sensitive areas like healthcare and finance, eventually incorporating sensitive information in building data-driven algorithms, it is vital to scrutinize whether these data face any privacy leakage risks. One potential threat arises from an adversary querying trained models using the public, non-sensitive attributes of entities in the training data to infer their private, sensitive attributes, a technique known as the attribute inference attack. This attack is particularly deceptive because, while it may perform poorly in predicting sensitive attributes across the entire dataset, it excels at predicting the sensitive attributes of records from a few vulnerable groups, a phenomenon known as disparate vulnerability. This paper illustrates that an adversary can take advantage of this disparity to carry out a series of new attacks, showcasing a threat level beyond previous imagination. We first develop a novel inference attack called the disparity inference attack, which targets the identification of high-risk groups within the dataset. We then introduce two targeted variations of the attribute inference attack that can identify and exploit a vulnerable subset of the training data, marking the first instances of targeted attacks in this category, achieving significantly higher accuracy than untargeted versions. We are also the first to introduce a novel and effective disparity mitigation technique that simultaneously preserves model performance and prevents any risk of targeted attacks.



## **34. Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective**

cs.CV

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2404.14309v3) [paper-pdf](http://arxiv.org/pdf/2404.14309v3)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Diffusion-Based Purification (DBP) has emerged as an effective defense mechanism against adversarial attacks. The success of DBP is often attributed to the forward diffusion process, which reduces the distribution gap between clean and adversarial images by adding Gaussian noise. While this explanation is theoretically sound, the exact role of this mechanism in enhancing robustness remains unclear. In this paper, through empirical analysis, we propose that the intrinsic stochasticity in the DBP process is the primary factor driving robustness. To test this hypothesis, we introduce a novel Deterministic White-Box (DW-box) setting to assess robustness in the absence of stochasticity, and we analyze attack trajectories and loss landscapes. Our results suggest that DBP models primarily rely on stochasticity to avoid effective attack directions, while their ability to purify adversarial perturbations may be limited. To further enhance the robustness of DBP models, we propose Adversarial Denoising Diffusion Training (ADDT), which incorporates classifier-guided adversarial perturbations into the diffusion training process, thereby strengthening the models' ability to purify adversarial perturbations. Additionally, we propose Rank-Based Gaussian Mapping (RBGM) to improve the compatibility of perturbations with diffusion models. Experimental results validate the effectiveness of ADDT. In conclusion, our study suggests that future research on DBP can benefit from a clearer distinction between stochasticity-driven and purification-driven robustness.



## **35. Commit-Reveal$^2$: Randomized Reveal Order Mitigates Last-Revealer Attacks in Commit-Reveal**

cs.CR

This paper will appear in the ICBC 2025 proceedings

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03936v1) [paper-pdf](http://arxiv.org/pdf/2504.03936v1)

**Authors**: Suheyon Lee, Euisin Gee

**Abstract**: Randomness generation is a fundamental component in blockchain systems, essential for tasks such as validator selection, zero-knowledge proofs, and decentralized finance operations. Traditional Commit-Reveal mechanisms provide simplicity and security but are susceptible to last revealer attacks, where an adversary can manipulate the random outcome by withholding their reveal. To address this vulnerability, we propose the Commit-Reveal$^2$ protocol, which employs a two-layer Commit-Reveal process to randomize the reveal order and mitigate the risk of such attacks. Additionally, we introduces a method to leverage off-chain networks to optimize communication costs and enhance efficiency. We implement a prototype of the proposed mechanism and publicly release the code to facilitate practical adoption and further research.



## **36. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

cs.LG

AAAI 2025

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2403.10045v4) [paper-pdf](http://arxiv.org/pdf/2403.10045v4)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information, so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks. Our implementation is available at: https://github.com/yumozi/GUARD.



## **37. SoK: Attacks on Modern Card Payments**

cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03363v1) [paper-pdf](http://arxiv.org/pdf/2504.03363v1)

**Authors**: Xenia Hofmeier, David Basin, Ralf Sasse, Jorge Toro-Pozo

**Abstract**: EMV is the global standard for smart card payments and its NFC-based version, EMV contactless, is widely used, also for mobile payments. In this systematization of knowledge, we examine attacks on the EMV contactless protocol. We provide a comprehensive framework encompassing its desired security properties and adversary models. We also identify and categorize a comprehensive collection of protocol flaws and show how different subsets thereof can be combined into attacks. In addition to this systematization, we examine the underlying reasons for the many attacks against EMV and point to a better way forward.



## **38. SLACK: Attacking LiDAR-based SLAM with Adversarial Point Injections**

cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03089v1) [paper-pdf](http://arxiv.org/pdf/2504.03089v1)

**Authors**: Prashant Kumar, Dheeraj Vattikonda, Kshitij Madhav Bhat, Kunal Dargan, Prem Kalra

**Abstract**: The widespread adoption of learning-based methods for the LiDAR makes autonomous vehicles vulnerable to adversarial attacks through adversarial \textit{point injections (PiJ)}. It poses serious security challenges for navigation and map generation. Despite its critical nature, no major work exists that studies learning-based attacks on LiDAR-based SLAM. Our work proposes SLACK, an end-to-end deep generative adversarial model to attack LiDAR scans with several point injections without deteriorating LiDAR quality. To facilitate SLACK, we design a novel yet simple autoencoder that augments contrastive learning with segmentation-based attention for precise reconstructions. SLACK demonstrates superior performance on the task of \textit{point injections (PiJ)} compared to the best baselines on KITTI and CARLA-64 dataset while maintaining accurate scan quality. We qualitatively and quantitatively demonstrate PiJ attacks using a fraction of LiDAR points. It severely degrades navigation and map quality without deteriorating the LiDAR scan quality.



## **39. Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning**

cs.CR

10 pages, 5 figures, research article, IEEE possible publication (in  submission)

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03077v1) [paper-pdf](http://arxiv.org/pdf/2504.03077v1)

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats.



## **40. Moving Target Defense Against Adversarial False Data Injection Attacks In Power Grids**

eess.SY

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03065v1) [paper-pdf](http://arxiv.org/pdf/2504.03065v1)

**Authors**: Yexiang Chen, Subhash Lakshminarayana, H. Vincent Poor

**Abstract**: Machine learning (ML)-based detectors have been shown to be effective in detecting stealthy false data injection attacks (FDIAs) that can bypass conventional bad data detectors (BDDs) in power systems. However, ML models are also vulnerable to adversarial attacks. A sophisticated perturbation signal added to the original BDD-bypassing FDIA can conceal the attack from ML-based detectors. In this paper, we develop a moving target defense (MTD) strategy to defend against adversarial FDIAs in power grids. We first develop an MTD-strengthened deep neural network (DNN) model, which deploys a pool of DNN models rather than a single static model that cooperate to detect the adversarial attack jointly. The MTD model pool introduces randomness to the ML model's decision boundary, thereby making the adversarial attacks detectable. Furthermore, to increase the effectiveness of the MTD strategy and reduce the computational costs associated with developing the MTD model pool, we combine this approach with the physics-based MTD, which involves dynamically perturbing the transmission line reactance and retraining the DNN-based detector to adapt to the new system topology. Simulations conducted on IEEE test bus systems demonstrate that the MTD-strengthened DNN achieves up to 94.2% accuracy in detecting adversarial FDIAs. When combined with a physics-based MTD, the detection accuracy surpasses 99%, while significantly reducing the computational costs of updating the DNN models. This approach requires only moderate perturbations to transmission line reactances, resulting in minimal increases in OPF cost.



## **41. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.09794v2) [paper-pdf](http://arxiv.org/pdf/2409.09794v2)

**Authors**: Hao Jian Huang, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using Raspberry Pi and Nvidia Jetson hardware by running the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, the testbed's capabilities are shown in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. The results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.



## **42. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.



## **43. A Study on Adversarial Robustness of Discriminative Prototypical Learning**

cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03782v1) [paper-pdf](http://arxiv.org/pdf/2504.03782v1)

**Authors**: Ramin Zarei Sabzevar, Hamed Mohammadzadeh, Tahmineh Tavakoli, Ahad Harati

**Abstract**: Deep neural networks demonstrate significant vulnerability to adversarial perturbations, posing risks for critical applications. Current adversarial training methods predominantly focus on robustness against attacks without explicitly leveraging geometric structures in the latent space, usually resulting in reduced accuracy on the original clean data. To address these issues, we propose a novel adversarial training framework named Adversarial Deep Positive-Negative Prototypes (Adv-DPNP), which integrates disriminative prototype-based learning with adversarial training. Adv-DPNP uses unified class prototypes serving dual roles as classifier weights and robust anchors, enhancing both intra-class compactness and inter-class separation in the latent space. Moreover, a novel dual-branch training mechanism maintains stable prototypes by updating them exclusively with clean data; while the feature extractor layers are learned using both clean and adversarial data to remain invariant against adversarial perturbations. In addition, our approach utilizes a composite loss function combining positive prototype alignment, negative prototype repulsion, and consistency regularization to further enhance discrimination, adversarial robustness, and clean accuracy. Extensive experiments conducted on standard benchmark datasets confirm the effectiveness of Adv-DPNP compared to state-of-the-art methods, achieving higher clean accuracy and competitive robustness under adversarial perturbations and common corruptions. Our code is available at https://github.com/fum-rpl/adv-dpnp



## **44. No Free Lunch with Guardrails**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.



## **45. Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems**

cs.LG

accepted by CVPR2025

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.00383v2) [paper-pdf](http://arxiv.org/pdf/2503.00383v2)

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Ling-Yu Duan, Alex C. Kot, Xudong Jiang

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.



## **46. Robust Unsupervised Domain Adaptation for 3D Point Cloud Segmentation Under Source Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.01659v2) [paper-pdf](http://arxiv.org/pdf/2504.01659v2)

**Authors**: Haosheng Li, Junjie Chen, Yuecong Xu, Kemi Ding

**Abstract**: Unsupervised domain adaptation (UDA) frameworks have shown good generalization capabilities for 3D point cloud semantic segmentation models on clean data. However, existing works overlook adversarial robustness when the source domain itself is compromised. To comprehensively explore the robustness of the UDA frameworks, we first design a stealthy adversarial point cloud generation attack that can significantly contaminate datasets with only minor perturbations to the point cloud surface. Based on that, we propose a novel dataset, AdvSynLiDAR, comprising synthesized contaminated LiDAR point clouds. With the generated corrupted data, we further develop the Adversarial Adaptation Framework (AAF) as the countermeasure. Specifically, by extending the key point sensitive (KPS) loss towards the Robust Long-Tail loss (RLT loss) and utilizing a decoder branch, our approach enables the model to focus on long-tail classes during the pre-training phase and leverages high-confidence decoded point cloud information to restore point cloud structures during the adaptation phase. We evaluated our AAF method on the AdvSynLiDAR dataset, where the results demonstrate that our AAF method can mitigate performance degradation under source adversarial perturbations for UDA in the 3D point cloud segmentation application.



## **47. Secure Generalization through Stochastic Bidirectional Parameter Updates Using Dual-Gradient Mechanism**

cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02213v1) [paper-pdf](http://arxiv.org/pdf/2504.02213v1)

**Authors**: Shourya Goel, Himanshi Tibrewal, Anant Jain, Anshul Pundhir, Pravendra Singh

**Abstract**: Federated learning (FL) has gained increasing attention due to privacy-preserving collaborative training on decentralized clients, mitigating the need to upload sensitive data to a central server directly. Nonetheless, recent research has underscored the risk of exposing private data to adversaries, even within FL frameworks. In general, existing methods sacrifice performance while ensuring resistance to privacy leakage in FL. We overcome these issues and generate diverse models at a global server through the proposed stochastic bidirectional parameter update mechanism. Using diverse models, we improved the generalization and feature representation in the FL setup, which also helped to improve the robustness of the model against privacy leakage without hurting the model's utility. We use global models from past FL rounds to follow systematic perturbation in parameter space at the server to ensure model generalization and resistance against privacy attacks. We generate diverse models (in close neighborhoods) for each client by using systematic perturbations in model parameters at a fine-grained level (i.e., altering each convolutional filter across the layers of the model) to improve the generalization and security perspective. We evaluated our proposed approach on four benchmark datasets to validate its superiority. We surpassed the state-of-the-art methods in terms of model utility and robustness towards privacy leakage. We have proven the effectiveness of our method by evaluating performance using several quantitative and qualitative results.



## **48. FairDAG: Consensus Fairness over Concurrent Causal Design**

cs.DB

17 pages, 15 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02194v1) [paper-pdf](http://arxiv.org/pdf/2504.02194v1)

**Authors**: Dakai Kang, Junchao Chen, Tien Tuan Anh Dinh, Mohammad Sadoghi

**Abstract**: The rise of cryptocurrencies like Bitcoin and Ethereum has driven interest in blockchain technology, with Ethereum's smart contracts enabling the growth of decentralized finance (DeFi). However, research has shown that adversaries exploit transaction ordering to extract profits through attacks like front-running, sandwich attacks, and liquidation manipulation. This issue affects both permissionless and permissioned blockchains, as block proposers have full control over transaction ordering. To address this, a more fair approach to transaction ordering is essential.   Existing fairness protocols, such as Pompe and Themis, operate on leader-based consensus protocols, which not only suffer from low throughput but also allow adversaries to manipulate transaction ordering. To address these limitations, we propose FairDAG-AB and FairDAG-RL, which leverage DAG-based consensus protocols.   We theoretically demonstrate that FairDAG protocols not only uphold fairness guarantees, as previous fairness protocols do, but also achieve higher throughput and greater resilience to adversarial ordering manipulation. Our deployment and evaluation on CloudLab further validate these claims.



## **49. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2503.21983v2) [paper-pdf](http://arxiv.org/pdf/2503.21983v2)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.



## **50. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



