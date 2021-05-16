from django import template

register = template.Library()

@register.simple_tag(takes_context=True)
def result_range(context):
    result_range = str(context['start']) + ":" + str(context['end'])
    return result_range

@register.simple_tag(takes_context=True)
def more_result(context):
    return context['end'] < len(context['result'])