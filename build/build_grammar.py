from tree_sitter import Language, Parser

if __name__ == '__main__':
    Language.build_library(
    # Store the library in the `build` directory
    'CodeSearchNet/function_parser/tree-sitter-languages.so',

    # Include one or more languages
        [
            '/home_local/guest-9530/Documents/program_augmentation/CodeSearchNet/build/tree-sitter-haskell',
            '/home_local/guest-9530/Documents/program_augmentation/CodeSearchNet/build/tree-sitter-python'
        ]
    )