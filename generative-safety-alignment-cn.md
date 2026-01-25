# 泛生成模型 - 安全对齐
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. M-ErasureBench: A Comprehensive Multimodal Evaluation Benchmark for Concept Erasure in Diffusion Models**

M-ErasureBench：扩散模型概念擦除的多模态综合评估基准 cs.CV

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22877v1) [paper-pdf](https://arxiv.org/pdf/2512.22877v1)

**Confidence**: 0.95

**Authors**: Ju-Hsuan Weng, Jia-Wei Liao, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Text-to-image diffusion models may generate harmful or copyrighted content, motivating research on concept erasure. However, existing approaches primarily focus on erasing concepts from text prompts, overlooking other input modalities that are increasingly critical in real-world applications such as image editing and personalized generation. These modalities can become attack surfaces, where erased concepts re-emerge despite defenses. To bridge this gap, we introduce M-ErasureBench, a novel multimodal evaluation framework that systematically benchmarks concept erasure methods across three input modalities: text prompts, learned embeddings, and inverted latents. For the latter two, we evaluate both white-box and black-box access, yielding five evaluation scenarios. Our analysis shows that existing methods achieve strong erasure performance against text prompts but largely fail under learned embeddings and inverted latents, with Concept Reproduction Rate (CRR) exceeding 90% in the white-box setting. To address these vulnerabilities, we propose IRECE (Inference-time Robustness Enhancement for Concept Erasure), a plug-and-play module that localizes target concepts via cross-attention and perturbs the associated latents during denoising. Experiments demonstrate that IRECE consistently restores robustness, reducing CRR by up to 40% under the most challenging white-box latent inversion scenario, while preserving visual quality. To the best of our knowledge, M-ErasureBench provides the first comprehensive benchmark of concept erasure beyond text prompts. Together with IRECE, our benchmark offers practical safeguards for building more reliable protective generative models.

摘要: 文本到图像扩散模型可能生成有害或受版权保护的内容，这推动了概念擦除技术的研究。然而，现有方法主要集中于从文本提示中擦除概念，忽视了在实际应用（如图像编辑和个性化生成）中日益重要的其他输入模态。这些模态可能成为攻击面，导致已擦除的概念绕过防御重新出现。为填补这一空白，我们提出了M-ErasureBench——一个新颖的多模态评估框架，系统性地在三种输入模态上对概念擦除方法进行基准测试：文本提示、学习嵌入和反转潜在向量。针对后两种模态，我们评估了白盒和黑盒访问权限，共形成五种评估场景。分析表明，现有方法在应对文本提示时表现出较强的擦除性能，但在学习嵌入和反转潜在向量场景下大多失效，其中白盒设置下的概念再现率（CRR）超过90%。为应对这些漏洞，我们提出了IRECE（推理时概念擦除鲁棒性增强模块），这是一个即插即用模块，通过交叉注意力定位目标概念并在去噪过程中扰动相关潜在向量。实验证明，IRECE能持续恢复鲁棒性，在最具挑战性的白盒潜在向量反转场景中将CRR降低达40%，同时保持视觉质量。据我们所知，M-ErasureBench首次提供了超越文本提示的全面概念擦除基准。结合IRECE，我们的基准为构建更可靠的安全生成模型提供了实用保障。



## **2. Wukong Framework for Not Safe For Work Detection in Text-to-Image systems**

Wukong框架：用于文生图系统中NSFW内容检测 cs.CV

Accepted by KDD'26 (round 1)

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2508.00591v2) [paper-pdf](https://arxiv.org/pdf/2508.00591v2)

**Confidence**: 0.95

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency.

摘要: 文生图（T2I）生成是一种流行的AI生成内容（AIGC）技术，能够实现多样化和创造性的图像合成。然而，部分输出可能包含不适合工作场所（NSFW）的内容（如暴力），违反社区准则。高效准确地检测NSFW内容（即外部防护机制）至关重要。现有外部防护机制分为两类：文本过滤器（分析用户提示但忽略T2I模型特定变体且易受对抗攻击）和图像过滤器（分析最终生成图像但计算成本高且引入延迟）。扩散模型作为现代T2I系统（如Stable Diffusion）的基础，通过使用包含ResNet和Transformer块的U-Net架构进行迭代去噪来生成图像。我们观察到：（1）早期去噪步骤定义了图像的语义布局；（2）U-Net中的交叉注意力层对文本与图像区域对齐至关重要。基于这些发现，我们提出Wukong——一个基于Transformer的NSFW检测框架，它利用早期去噪步骤的中间输出并重用U-Net预训练的交叉注意力参数。Wukong在扩散过程中运行，无需等待完整图像生成即可实现早期检测。我们还引入了一个包含提示词、种子和图像特定NSFW标签的新数据集，并在此数据集及两个公共基准上评估Wukong。结果表明，Wukong显著优于基于文本的防护机制，并达到与图像过滤器相当的准确率，同时提供更高的效率。



## **3. SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models**

SafeRedir：基于提示嵌入重定向的图像生成模型鲁棒遗忘方法 cs.CV

Code at https://github.com/ryliu68/SafeRedir

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08623v1) [paper-pdf](https://arxiv.org/pdf/2601.08623v1)

**Confidence**: 0.95

**Authors**: Renyang Liu, Kangjie Chen, Han Qiu, Jie Zhang, Kwok-Yan Lam, Tianwei Zhang, See-Kiong Ng

**Abstract**: Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at https://github.com/ryliu68/SafeRedir.

摘要: 图像生成模型（IGMs）虽然能够生成令人印象深刻的创意内容，但通常会从其训练数据中记忆大量不良概念，导致生成不安全内容（如NSFW图像和受版权保护的艺术风格）。这些行为在实际部署中构成持续的安全与合规风险，且由于后处理过滤机制的鲁棒性有限及缺乏细粒度语义控制，无法通过此类方法可靠缓解。近期遗忘方法试图在模型层面消除有害概念，但存在需要昂贵重训练、降低良性生成质量或无法抵御提示改写和对抗攻击等局限。为应对这些挑战，我们提出SafeRedir——一种通过提示嵌入重定向实现鲁棒遗忘的轻量级推理框架。该方法无需修改底层IGMs，通过在嵌入空间进行词元级干预，自适应地将不安全提示重定向至安全语义区域。框架包含两个核心组件：用于识别不安全生成轨迹的潜在感知多模态安全分类器，以及配备词元掩码预测器和自适应缩放预测器的词元级增量生成器，用于实现精准语义重定向并定位调节干预过程。在多个代表性遗忘任务上的实验结果表明，SafeRedir具备有效的遗忘能力、高语义与感知保真度、鲁棒的图像质量及增强的抗对抗攻击能力。此外，该方法能有效泛化至多种扩散骨干网络和现有遗忘模型，验证了其即插即用兼容性与广泛适用性。代码与数据详见：https://github.com/ryliu68/SafeRedir。



## **4. ActErase: A Training-Free Paradigm for Precise Concept Erasure via Activation Patching**

ActErase：一种基于激活修补的免训练精确概念擦除范式 cs.CV

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00267v1) [paper-pdf](https://arxiv.org/pdf/2601.00267v1)

**Confidence**: 0.95

**Authors**: Yi Sun, Xinhao Zhong, Hongyan Li, Yimin Zhou, Junhao Li, Bin Chen, Xuan Wang

**Abstract**: Recent advances in text-to-image diffusion models have demonstrated remarkable generation capabilities, yet they raise significant concerns regarding safety, copyright, and ethical implications. Existing concept erasure methods address these risks by removing sensitive concepts from pre-trained models, but most of them rely on data-intensive and computationally expensive fine-tuning, which poses a critical limitation. To overcome these challenges, inspired by the observation that the model's activations are predominantly composed of generic concepts, with only a minimal component can represent the target concept, we propose a novel training-free method (ActErase) for efficient concept erasure. Specifically, the proposed method operates by identifying activation difference regions via prompt-pair analysis, extracting target activations and dynamically replacing input activations during forward passes. Comprehensive evaluations across three critical erasure tasks (nudity, artistic style, and object removal) demonstrates that our training-free method achieves state-of-the-art (SOTA) erasure performance, while effectively preserving the model's overall generative capability. Our approach also exhibits strong robustness against adversarial attacks, establishing a new plug-and-play paradigm for lightweight yet effective concept manipulation in diffusion models.

摘要: 文本到图像扩散模型的最新进展展现了卓越的生成能力，但也引发了安全、版权和伦理方面的重大关切。现有概念擦除方法通过从预训练模型中移除敏感概念来应对这些风险，但大多依赖数据密集且计算成本高昂的微调，存在显著局限性。为克服这些挑战，受模型激活主要由通用概念构成、仅极小部分能表示目标概念这一观察启发，我们提出一种新颖的免训练方法（ActErase）以实现高效概念擦除。具体而言，该方法通过提示对分析识别激活差异区域，提取目标激活并在前向传播过程中动态替换输入激活。在三个关键擦除任务（裸露内容、艺术风格和对象移除）上的综合评估表明，我们的免训练方法实现了最先进的擦除性能，同时有效保持了模型的整体生成能力。该方法还展现出对抗攻击的强鲁棒性，为扩散模型中的轻量级高效概念操作建立了即插即用的新范式。



## **5. PEPPER: Perception-Guided Perturbation for Robust Backdoor Defense in Text-to-Image Diffusion Models**

PEPPER：面向文本到图像扩散模型鲁棒后门防御的感知引导扰动方法 cs.CL

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2511.16830v2) [paper-pdf](https://arxiv.org/pdf/2511.16830v2)

**Confidence**: 0.95

**Authors**: Oscar Chew, Po-Yi Lu, Jayden Lin, Kuan-Hao Huang, Hsuan-Tien Lin

**Abstract**: Recent studies show that text to image (T2I) diffusion models are vulnerable to backdoor attacks, where a trigger in the input prompt can steer generation toward harmful or unintended content. To address this, we introduce PEPPER (PErcePtion Guided PERturbation), a backdoor defense that rewrites the caption into a semantically distant yet visually similar caption while adding unobstructive elements. With this rewriting strategy, PEPPER disrupt the trigger embedded in the input prompt, dilute the influence of trigger tokens and thereby achieve enhanced robustness. Experiments show that PEPPER is particularly effective against text encoder based attacks, substantially reducing attack success while preserving generation quality. Beyond this, PEPPER can be paired with any existing defenses yielding consistently stronger and generalizable robustness than any standalone method. Our code will be released on Github.

摘要: 近期研究表明，文本到图像（T2I）扩散模型易受后门攻击，输入提示中的触发器可能引导生成有害或非预期内容。为此，我们提出PEPPER（感知引导扰动）——一种通过将输入描述重写为语义疏远但视觉相似的描述，并添加非干扰性元素的后门防御方法。该重写策略能破坏输入提示中嵌入的触发器，稀释触发词的影响，从而提升鲁棒性。实验表明，PEPPER对基于文本编码器的攻击尤为有效，在保持生成质量的同时显著降低攻击成功率。此外，PEPPER可与现有防御方法结合使用，相比单一方法能持续产生更强且可泛化的鲁棒性。代码将在GitHub开源。



## **6. SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders**

SAEmnesia：利用监督稀疏自编码器消除扩散模型中的概念 cs.CV

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2509.21379v2) [paper-pdf](https://arxiv.org/pdf/2509.21379v2)

**Confidence**: 0.95

**Authors**: Enrico Cassano, Riccardo Renzulli, Marco Nurisso, Mirko Zaffaroni, Alan Perotti, Marco Grangetto

**Abstract**: Concept unlearning in diffusion models is hampered by feature splitting, where concepts are distributed across many latent features, making their removal challenging and computationally expensive. We introduce SAEmnesia, a supervised sparse autoencoder framework that overcomes this by enforcing one-to-one concept-neuron mappings. By systematically labeling concepts during training, our method achieves feature centralization, binding each concept to a single, interpretable neuron. This enables highly targeted and efficient concept erasure. SAEmnesia reduces hyperparameter search by 96.7% and achieves a 9.2% improvement over the state-of-the-art on the UnlearnCanvas benchmark. Our method also demonstrates superior scalability in sequential unlearning, improving accuracy by 28.4% when removing nine objects, establishing a new standard for precise and controllable concept erasure. Moreover, SAEmnesia mitigates the possibility of generating unwanted content under adversarial attack and effectively removes nudity when evaluated with I2P.

摘要: 扩散模型中的概念遗忘受限于特征分裂问题，即概念分散于多个潜在特征中，导致其移除既困难又计算成本高昂。我们提出了SAEmnesia，一种监督稀疏自编码器框架，通过强制实施一对一的概念-神经元映射来克服此问题。通过在训练过程中系统性地标记概念，我们的方法实现了特征集中化，将每个概念绑定到单一、可解释的神经元上。这使得概念擦除具有高度针对性和高效性。SAEmnesia将超参数搜索减少了96.7%，并在UnlearnCanvas基准测试中实现了比现有最佳方法9.2%的性能提升。我们的方法在顺序遗忘任务中也展现出卓越的可扩展性，当移除九个对象时，准确率提高了28.4%，为精确可控的概念擦除设立了新标准。此外，SAEmnesia降低了在对抗攻击下生成不良内容的可能性，并在使用I2P评估时有效移除了裸露内容。



## **7. Beyond the Safety Tax: Mitigating Unsafe Text-to-Image Generation via External Safety Rectification**

超越安全税：通过外部安全校正缓解不安全的文本到图像生成 cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2508.21099v3) [paper-pdf](https://arxiv.org/pdf/2508.21099v3)

**Confidence**: 0.95

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Text-to-image (T2I) generative models have achieved remarkable visual fidelity, yet remain vulnerable to generating unsafe content. Existing safety defenses typically intervene internally within the generative model, but suffer from severe concept entanglement, leading to degradation of benign generation quality, a trade-off we term the Safety Tax. To overcome this limitation, we advocate a paradigm shift from destructive internal editing to external safety rectification. Following this principle, we propose SafePatch, a structurally isolated safety module that performs external, interpretable rectification without modifying the base model. The core backbone of SafePatch is architecturally instantiated as a trainable clone of the base model's encoder, allowing it to inherit rich semantic priors and maintain representation consistency. To enable interpretable safety rectification, we construct a strictly aligned counterfactual safety dataset (ACS) for differential supervision training. Across nudity and multi-category benchmarks and recent adversarial prompt attacks, SafePatch achieves robust unsafe suppression (7% unsafe on I2P) while preserving image quality and semantic alignment.

摘要: 文本到图像（T2I）生成模型已实现显著的视觉保真度，但仍易生成不安全内容。现有安全防御通常干预生成模型内部，但存在严重概念纠缠问题，导致良性生成质量下降——我们称这种权衡为“安全税”。为克服此限制，我们倡导从破坏性内部编辑转向外部安全校正的范式转变。基于此原则，我们提出SafePatch，这是一个结构隔离的安全模块，可在不修改基础模型的情况下执行外部可解释校正。SafePatch的核心架构实现为基础模型编码器的可训练克隆，使其能够继承丰富的语义先验并保持表示一致性。为实现可解释的安全校正，我们构建了严格对齐的反事实安全数据集（ACS）用于差分监督训练。在裸露内容和多类别基准测试及近期对抗性提示攻击中，SafePatch实现了稳健的不安全内容抑制（I2P数据集上不安全率7%），同时保持了图像质量和语义对齐。



## **8. Synthetic Voices, Real Threats: Evaluating Large Text-to-Speech Models in Generating Harmful Audio**

合成语音，真实威胁：评估大型文本转语音模型生成有害音频的能力 cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10913v1) [paper-pdf](https://arxiv.org/pdf/2511.10913v1)

**Confidence**: 0.95

**Authors**: Guangke Chen, Yuhui Wang, Shouling Ji, Xiapu Luo, Ting Wang

**Abstract**: Modern text-to-speech (TTS) systems, particularly those built on Large Audio-Language Models (LALMs), generate high-fidelity speech that faithfully reproduces input text and mimics specified speaker identities. While prior misuse studies have focused on speaker impersonation, this work explores a distinct content-centric threat: exploiting TTS systems to produce speech containing harmful content. Realizing such threats poses two core challenges: (1) LALM safety alignment frequently rejects harmful prompts, yet existing jailbreak attacks are ill-suited for TTS because these systems are designed to faithfully vocalize any input text, and (2) real-world deployment pipelines often employ input/output filters that block harmful text and audio.   We present HARMGEN, a suite of five attacks organized into two families that address these challenges. The first family employs semantic obfuscation techniques (Concat, Shuffle) that conceal harmful content within text. The second leverages audio-modality exploits (Read, Spell, Phoneme) that inject harmful content through auxiliary audio channels while maintaining benign textual prompts. Through evaluation across five commercial LALMs-based TTS systems and three datasets spanning two languages, we demonstrate that our attacks substantially reduce refusal rates and increase the toxicity of generated speech.   We further assess both reactive countermeasures deployed by audio-streaming platforms and proactive defenses implemented by TTS providers. Our analysis reveals critical vulnerabilities: deepfake detectors underperform on high-fidelity audio; reactive moderation can be circumvented by adversarial perturbations; while proactive moderation detects 57-93% of attacks. Our work highlights a previously underexplored content-centric misuse vector for TTS and underscore the need for robust cross-modal safeguards throughout training and deployment.

摘要: 现代文本转语音（TTS）系统，特别是基于大型音频-语言模型（LALMs）构建的系统，能够生成高保真语音，忠实复现输入文本并模仿指定说话人身份。以往滥用研究多集中于说话人身份伪造，而本研究探索了一种以内容为中心的新型威胁：利用TTS系统生成包含有害内容的语音。实现此类威胁面临两大核心挑战：（1）LALMs的安全对齐机制常拒绝有害提示，但现有越狱攻击不适用于TTS系统，因为这些系统被设计为忠实语音化任何输入文本；（2）实际部署流程通常采用输入/输出过滤器来拦截有害文本和音频。

我们提出HARMGEN攻击套件，包含五大攻击方法，分为两类应对上述挑战。第一类采用语义混淆技术（拼接、重排），将有害内容隐藏于文本中；第二类利用音频模态漏洞（朗读、拼写、音素），通过辅助音频通道注入有害内容，同时保持文本提示的良性。通过对五种基于LALMs的商业TTS系统和涵盖两种语言的三个数据集进行评估，我们证明这些攻击能显著降低系统拒绝率并提升生成语音的毒性。

我们进一步评估了音频流平台部署的被动防御措施和TTS提供商实施的主动防御机制。分析揭示关键漏洞：深度伪造检测器对高保真音频性能不足；被动审核可通过对抗性扰动规避；而主动审核能检测57-93%的攻击。本研究凸显了TTS系统中先前未被充分探索的内容中心滥用向量，并强调在训练和部署全过程中建立鲁棒跨模态防护机制的必要性。



## **9. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

AEIOU：针对文本到图像模型中NSFW提示的统一防御框架 cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2412.18123v3) [paper-pdf](https://arxiv.org/pdf/2412.18123v3)

**Confidence**: 0.95

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Tong Zhang, Rui Zeng, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose AEIOU, a defense framework that is adaptable, efficient, interpretable, optimizable, and unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.

摘要: 随着文本到图像（T2I）模型的进步和广泛应用，其相关的安全问题日益凸显。恶意用户利用这些模型，通过有害或对抗性提示生成不适合工作场所（NSFW）的图像，这凸显了需要有效的安全措施来确保模型输出的完整性和合规性。然而，现有的检测方法往往准确率低且效率不高。本文提出AEIOU，一个针对T2I模型中NSFW提示的适应性、高效性、可解释性、可优化性和统一性的防御框架。AEIOU从模型文本编码器的隐藏状态中提取NSFW特征，利用这些特征的可分离性来检测NSFW提示。检测过程高效，仅需极少的推理时间。AEIOU还提供结果的实时解释，并支持通过数据增强技术进行优化。该框架具有通用性，可适配多种T2I架构。我们的大量实验表明，AEIOU显著优于商业和开源的内容审核工具，在所有数据集上准确率超过95%，效率提升至少十倍。它能有效应对自适应攻击，并在少样本和多标签场景中表现优异。



