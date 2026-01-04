"""
Text templates for sentiment analysis tasks.
These templates help the model understand sentiment-related contexts.
"""

# Templates for sentiment classification
SENTIMENT_TEMPLATES = [
    "a photo expressing {} sentiment.",
    "an image showing {} emotion.",
    "a {} post on social media.",
    "a picture with {} feeling.",
    "social media content that is {}.",
    "an image that conveys {} mood.",
    "a {} scene.",
    "a photo that makes people feel {}.",
    "content with {} tone.",
    "a {} message.",
]

# Simple templates for quick experiments
SIMPLE_SENTIMENT_TEMPLATES = [
    "a photo of {} sentiment.",
    "an image of {} emotion.",
    "a {} post.",
]

# Detailed templates for better context understanding
DETAILED_SENTIMENT_TEMPLATES = [
    "a photo expressing {} sentiment on social media.",
    "an image showing {} emotion shared by users.",
    "a {} post with text and image on social media.",
    "a picture with {} feeling posted online.",
    "social media content that conveys {} sentiment.",
    "an image with {} emotional tone from social networks.",
    "a {} scene captured and shared online.",
    "a photo that makes people feel {} when viewing.",
    "content with {} sentiment posted by social media users.",
    "a {} message combining image and text.",
    "a social media post reflecting {} mood.",
    "an image-text pair expressing {} feelings.",
]

# Templates specifically for multimodal sentiment
MULTIMODAL_SENTIMENT_TEMPLATES = [
    "a photo and text expressing {} sentiment.",
    "an image with caption showing {} emotion.",
    "a multimodal post that is {}.",
    "a picture with text conveying {} feeling.",
    "combined image and text content that is {}.",
]

# Templates for each sentiment class with specific descriptions
SENTIMENT_CLASS_DESCRIPTIONS = {
    "positive": [
        "happy and cheerful",
        "joyful and delighted",
        "pleased and satisfied",
        "optimistic and hopeful",
        "excited and enthusiastic",
    ],
    "negative": [
        "sad and unhappy",
        "angry and frustrated",
        "disappointed and upset",
        "worried and anxious",
        "depressed and gloomy",
    ],
    "neutral": [
        "calm and balanced",
        "objective and factual",
        "matter-of-fact",
        "unemotional",
        "informative",
    ],
}


def get_sentiment_templates(template_type="default"):
    """
    Get sentiment templates based on type.
    
    Args:
        template_type: Type of templates to use
            - "default": Standard sentiment templates
            - "simple": Simple and concise templates
            - "detailed": Detailed and context-rich templates
            - "multimodal": Templates specifically for multimodal content
    
    Returns:
        List of template strings
    """
    if template_type == "simple":
        return SIMPLE_SENTIMENT_TEMPLATES
    elif template_type == "detailed":
        return DETAILED_SENTIMENT_TEMPLATES
    elif template_type == "multimodal":
        return MULTIMODAL_SENTIMENT_TEMPLATES
    else:
        return SENTIMENT_TEMPLATES

