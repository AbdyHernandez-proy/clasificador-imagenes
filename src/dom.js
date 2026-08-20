export function requireElementById(id) {
    const element = document.getElementById(id);

    if (!element) {
        throw new Error(`Elemento requerido no encontrado: #${id}`);
    }

    return element;
}

export function requireSelector(selector) {
    const element = document.querySelector(selector);

    if (!element) {
        throw new Error(`Elemento requerido no encontrado: ${selector}`);
    }

    return element;
}
