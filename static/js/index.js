const select = document.querySelector('.form-select')
const form = document.querySelector('.geo-form')

function eventHandler(event) {
    form.submit()
}

select.addEventListener('change', eventHandler)