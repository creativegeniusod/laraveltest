<?php

namespace App\Http\Requests\Media;
use App\Http\Requests\CoreRequest;
use Illuminate\Foundation\Http\FormRequest;

class StoreMedia extends CoreRequest
{
    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {
        return [
            'image' => 'required|image|mimes:jpeg,png,jpg,gif|max:2048'
        ];
    }
}
