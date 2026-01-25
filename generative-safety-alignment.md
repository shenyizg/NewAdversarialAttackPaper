# General Generative Models - Safety Alignment
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. M-ErasureBench: A Comprehensive Multimodal Evaluation Benchmark for Concept Erasure in Diffusion Models**

cs.CV

**SubmitDate**: 2025-12-28    [abs](http://arxiv.org/abs/2512.22877v1) [paper-pdf](https://arxiv.org/pdf/2512.22877v1)

**Confidence**: 0.95

**Authors**: Ju-Hsuan Weng, Jia-Wei Liao, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Text-to-image diffusion models may generate harmful or copyrighted content, motivating research on concept erasure. However, existing approaches primarily focus on erasing concepts from text prompts, overlooking other input modalities that are increasingly critical in real-world applications such as image editing and personalized generation. These modalities can become attack surfaces, where erased concepts re-emerge despite defenses. To bridge this gap, we introduce M-ErasureBench, a novel multimodal evaluation framework that systematically benchmarks concept erasure methods across three input modalities: text prompts, learned embeddings, and inverted latents. For the latter two, we evaluate both white-box and black-box access, yielding five evaluation scenarios. Our analysis shows that existing methods achieve strong erasure performance against text prompts but largely fail under learned embeddings and inverted latents, with Concept Reproduction Rate (CRR) exceeding 90% in the white-box setting. To address these vulnerabilities, we propose IRECE (Inference-time Robustness Enhancement for Concept Erasure), a plug-and-play module that localizes target concepts via cross-attention and perturbs the associated latents during denoising. Experiments demonstrate that IRECE consistently restores robustness, reducing CRR by up to 40% under the most challenging white-box latent inversion scenario, while preserving visual quality. To the best of our knowledge, M-ErasureBench provides the first comprehensive benchmark of concept erasure beyond text prompts. Together with IRECE, our benchmark offers practical safeguards for building more reliable protective generative models.



## **2. Wukong Framework for Not Safe For Work Detection in Text-to-Image systems**

cs.CV

Accepted by KDD'26 (round 1)

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2508.00591v2) [paper-pdf](https://arxiv.org/pdf/2508.00591v2)

**Confidence**: 0.95

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency.



## **3. SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models**

cs.CV

Code at https://github.com/ryliu68/SafeRedir

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08623v1) [paper-pdf](https://arxiv.org/pdf/2601.08623v1)

**Confidence**: 0.95

**Authors**: Renyang Liu, Kangjie Chen, Han Qiu, Jie Zhang, Kwok-Yan Lam, Tianwei Zhang, See-Kiong Ng

**Abstract**: Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at https://github.com/ryliu68/SafeRedir.



## **4. ActErase: A Training-Free Paradigm for Precise Concept Erasure via Activation Patching**

cs.CV

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00267v1) [paper-pdf](https://arxiv.org/pdf/2601.00267v1)

**Confidence**: 0.95

**Authors**: Yi Sun, Xinhao Zhong, Hongyan Li, Yimin Zhou, Junhao Li, Bin Chen, Xuan Wang

**Abstract**: Recent advances in text-to-image diffusion models have demonstrated remarkable generation capabilities, yet they raise significant concerns regarding safety, copyright, and ethical implications. Existing concept erasure methods address these risks by removing sensitive concepts from pre-trained models, but most of them rely on data-intensive and computationally expensive fine-tuning, which poses a critical limitation. To overcome these challenges, inspired by the observation that the model's activations are predominantly composed of generic concepts, with only a minimal component can represent the target concept, we propose a novel training-free method (ActErase) for efficient concept erasure. Specifically, the proposed method operates by identifying activation difference regions via prompt-pair analysis, extracting target activations and dynamically replacing input activations during forward passes. Comprehensive evaluations across three critical erasure tasks (nudity, artistic style, and object removal) demonstrates that our training-free method achieves state-of-the-art (SOTA) erasure performance, while effectively preserving the model's overall generative capability. Our approach also exhibits strong robustness against adversarial attacks, establishing a new plug-and-play paradigm for lightweight yet effective concept manipulation in diffusion models.



## **5. PEPPER: Perception-Guided Perturbation for Robust Backdoor Defense in Text-to-Image Diffusion Models**

cs.CL

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2511.16830v2) [paper-pdf](https://arxiv.org/pdf/2511.16830v2)

**Confidence**: 0.95

**Authors**: Oscar Chew, Po-Yi Lu, Jayden Lin, Kuan-Hao Huang, Hsuan-Tien Lin

**Abstract**: Recent studies show that text to image (T2I) diffusion models are vulnerable to backdoor attacks, where a trigger in the input prompt can steer generation toward harmful or unintended content. To address this, we introduce PEPPER (PErcePtion Guided PERturbation), a backdoor defense that rewrites the caption into a semantically distant yet visually similar caption while adding unobstructive elements. With this rewriting strategy, PEPPER disrupt the trigger embedded in the input prompt, dilute the influence of trigger tokens and thereby achieve enhanced robustness. Experiments show that PEPPER is particularly effective against text encoder based attacks, substantially reducing attack success while preserving generation quality. Beyond this, PEPPER can be paired with any existing defenses yielding consistently stronger and generalizable robustness than any standalone method. Our code will be released on Github.



## **6. SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders**

cs.CV

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2509.21379v2) [paper-pdf](https://arxiv.org/pdf/2509.21379v2)

**Confidence**: 0.95

**Authors**: Enrico Cassano, Riccardo Renzulli, Marco Nurisso, Mirko Zaffaroni, Alan Perotti, Marco Grangetto

**Abstract**: Concept unlearning in diffusion models is hampered by feature splitting, where concepts are distributed across many latent features, making their removal challenging and computationally expensive. We introduce SAEmnesia, a supervised sparse autoencoder framework that overcomes this by enforcing one-to-one concept-neuron mappings. By systematically labeling concepts during training, our method achieves feature centralization, binding each concept to a single, interpretable neuron. This enables highly targeted and efficient concept erasure. SAEmnesia reduces hyperparameter search by 96.7% and achieves a 9.2% improvement over the state-of-the-art on the UnlearnCanvas benchmark. Our method also demonstrates superior scalability in sequential unlearning, improving accuracy by 28.4% when removing nine objects, establishing a new standard for precise and controllable concept erasure. Moreover, SAEmnesia mitigates the possibility of generating unwanted content under adversarial attack and effectively removes nudity when evaluated with I2P.



## **7. Beyond the Safety Tax: Mitigating Unsafe Text-to-Image Generation via External Safety Rectification**

cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2508.21099v3) [paper-pdf](https://arxiv.org/pdf/2508.21099v3)

**Confidence**: 0.95

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Text-to-image (T2I) generative models have achieved remarkable visual fidelity, yet remain vulnerable to generating unsafe content. Existing safety defenses typically intervene internally within the generative model, but suffer from severe concept entanglement, leading to degradation of benign generation quality, a trade-off we term the Safety Tax. To overcome this limitation, we advocate a paradigm shift from destructive internal editing to external safety rectification. Following this principle, we propose SafePatch, a structurally isolated safety module that performs external, interpretable rectification without modifying the base model. The core backbone of SafePatch is architecturally instantiated as a trainable clone of the base model's encoder, allowing it to inherit rich semantic priors and maintain representation consistency. To enable interpretable safety rectification, we construct a strictly aligned counterfactual safety dataset (ACS) for differential supervision training. Across nudity and multi-category benchmarks and recent adversarial prompt attacks, SafePatch achieves robust unsafe suppression (7% unsafe on I2P) while preserving image quality and semantic alignment.



## **8. Synthetic Voices, Real Threats: Evaluating Large Text-to-Speech Models in Generating Harmful Audio**

cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10913v1) [paper-pdf](https://arxiv.org/pdf/2511.10913v1)

**Confidence**: 0.95

**Authors**: Guangke Chen, Yuhui Wang, Shouling Ji, Xiapu Luo, Ting Wang

**Abstract**: Modern text-to-speech (TTS) systems, particularly those built on Large Audio-Language Models (LALMs), generate high-fidelity speech that faithfully reproduces input text and mimics specified speaker identities. While prior misuse studies have focused on speaker impersonation, this work explores a distinct content-centric threat: exploiting TTS systems to produce speech containing harmful content. Realizing such threats poses two core challenges: (1) LALM safety alignment frequently rejects harmful prompts, yet existing jailbreak attacks are ill-suited for TTS because these systems are designed to faithfully vocalize any input text, and (2) real-world deployment pipelines often employ input/output filters that block harmful text and audio.   We present HARMGEN, a suite of five attacks organized into two families that address these challenges. The first family employs semantic obfuscation techniques (Concat, Shuffle) that conceal harmful content within text. The second leverages audio-modality exploits (Read, Spell, Phoneme) that inject harmful content through auxiliary audio channels while maintaining benign textual prompts. Through evaluation across five commercial LALMs-based TTS systems and three datasets spanning two languages, we demonstrate that our attacks substantially reduce refusal rates and increase the toxicity of generated speech.   We further assess both reactive countermeasures deployed by audio-streaming platforms and proactive defenses implemented by TTS providers. Our analysis reveals critical vulnerabilities: deepfake detectors underperform on high-fidelity audio; reactive moderation can be circumvented by adversarial perturbations; while proactive moderation detects 57-93% of attacks. Our work highlights a previously underexplored content-centric misuse vector for TTS and underscore the need for robust cross-modal safeguards throughout training and deployment.



## **9. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

cs.CR

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2412.18123v3) [paper-pdf](https://arxiv.org/pdf/2412.18123v3)

**Confidence**: 0.95

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Tong Zhang, Rui Zeng, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models advance and gain widespread adoption, their associated safety concerns are becoming increasingly critical. Malicious users exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, underscoring the need for effective safeguards to ensure the integrity and compliance of model outputs. However, existing detection methods often exhibit low accuracy and inefficiency. In this paper, we propose AEIOU, a defense framework that is adaptable, efficient, interpretable, optimizable, and unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95\% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.



